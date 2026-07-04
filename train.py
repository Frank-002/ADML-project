import argparse
import math
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from utils.model_builder import build_model_and_preprocess

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.SPairDataset import SPairDataset
from utils.evaluator import evaluate_one_epoch
from utils.trainer import train_one_epoch

# Largest per-forward batch that fits in ~12 GB of VRAM in fp32, sized for
# --unfreeze-layers up to 5; doubled when --amp halves the
# activations, or overridden with --real-batch on other GPUs. The effective
# batch is reached via gradient accumulation on top of it.
MAX_REAL_BATCH = {"DINOV2": 8, "DINOV3": 4, "SAM": 2}


def parse_args():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    # W&B sweep (hyperparameter search on SPair small, runs before the actual training)
    common.add_argument("--skip-sweep", action="store_true", help="skip the hyperparameter search and use the CLI hyperparameters directly")
    common.add_argument("--sweep-id", type=str, default=None, help="id of an existing sweep to join")
    common.add_argument("--sweep-count", type=int, default=12, help="number of sweep runs")

    # Hyperparameters: lr / tau / effective-batch are proposed by W&B during
    # the sweep; the CLI values are used only with --skip-sweep
    common.add_argument("--max-epochs", type=int, default=20, help="maximum number of epochs (also defines the cosine T_max)")
    common.add_argument("--patience", type=int, default=3, help="early stopping: epochs without val PCK@0.10 improvement")
    common.add_argument("--lr", type=float, default=1e-5)
    common.add_argument("--unfreeze-layers", type=int, required=True, help="blocks to unfreeze starting from the head, fixed for the whole pipeline (sweep + final training); relaunch with a different value to compare")
    common.add_argument("--cosine-decay", type=float, default=0.01, help="final lr = lr * cosine_decay")
    common.add_argument("--tau", type=float, default=0.05, help="InfoNCE temperature")
    common.add_argument("--effective-batch", type=int, default=32, help="effective training batch, reached via gradient accumulation (validation always uses 1)")
    common.add_argument("--real-batch", type=int, default=None, help="per-forward batch size; defaults to MAX_REAL_BATCH for the model")
    common.add_argument("--no-amp", dest="amp", action="store_false", help="disable bf16 mixed precision and train in fp32 (default: amp on, validation always fp32)")
    common.add_argument("--compile", action="store_true", help="torch.compile the backbone forward (~10-30%% faster steps; needs Triton, on Windows: pip install triton-windows)")
    common.add_argument("--weight-decay", type=float, default=0.01)
    common.add_argument("--max-grad-norm", type=float, default=1.0)
    common.add_argument("--num-workers", type=int, default=4, help="dataloader workers; raise it if GPU utilization is spiky, lower it if system RAM fills up (train+val keep 2x this many worker processes alive)")
    common.add_argument("--wandb-project", type=str, default="ADML-project")

    parser = argparse.ArgumentParser()

    model = parser.add_subparsers(dest='model', required=True)
    model.add_parser("DINOV2", parents=[common])
    model.add_parser("DINOV3", parents=[common])
    model.add_parser("SAM", parents=[common])

    return parser.parse_args()


def get_backbone(model, model_name) -> torch.nn.Module:
    if model_name == "SAM":
        return model.model.model  # SamPredictor -> Sam
    return model.model


def compile_backbone(backbone, model_name) -> None:
    """
    torch.compile del percorso pesante del forward, in-place: niente wrapper
    OptimizedModule sul modulo, cosi' le chiavi dello state_dict (e quindi i
    checkpoint letti da eval.py) restano identiche.
    """
    if model_name in ("DINOV2", "DINOV3"):
        # I wrapper chiamano forward_features, non forward: si compila il metodo
        backbone.forward_features = torch.compile(backbone.forward_features)
    elif model_name == "SAM":
        backbone.image_encoder.compile()
    else:
        raise NotImplementedError


def unfreeze_last_layers(model, model_name, num_layers) -> list:
    """
    Freeze the whole backbone and re-enable only the last num_layers blocks
    (starting from the head) plus the final modules (norm / neck).

    return: list of trainable parameters for the optimizer
    """
    backbone = get_backbone(model, model_name)

    for p in backbone.parameters():
        p.requires_grad_(False)

    if model_name in ("DINOV2", "DINOV3"):
        blocks = backbone.blocks
        tail = list(blocks[len(blocks) - num_layers:]) + [backbone.norm]
    elif model_name == "SAM":
        encoder = backbone.image_encoder
        blocks = encoder.blocks
        tail = list(blocks[len(blocks) - num_layers:]) + [encoder.neck]
    else:
        raise NotImplementedError

    for module in tail:
        for p in module.parameters():
            p.requires_grad_(True)

    return [p for p in backbone.parameters() if p.requires_grad]


def collate_train(batch):
    """
    Collate for training with batch_size > 1.

    Images are already padded to a fixed square by PreProcess, so they stack
    directly. Keypoints vary in number per pair (SPair), so they are padded to
    the batch max and a boolean kps_valid_mask marks the real ones. Only the
    fields consumed by dense_infonce_loss are kept.
    """
    src_img = torch.stack([b["src_img"] for b in batch])
    trg_img = torch.stack([b["trg_img"] for b in batch])
    src_nopad_size = torch.stack([b["src_nopad_size"] for b in batch])
    trg_nopad_size = torch.stack([b["trg_nopad_size"] for b in batch])

    B = len(batch)
    max_k = max(b["src_kps"].shape[0] for b in batch)

    src_kps = torch.full((B, max_k, 2), -1.0)
    trg_kps = torch.full((B, max_k, 2), -1.0)
    for i, b in enumerate(batch):
        k = b["src_kps"].shape[0]
        src_kps[i, :k] = b["src_kps"]
        trg_kps[i, :k] = b["trg_kps"]

    # Valid where both src and trg keypoints are set: excludes the -1 padding
    # (and any -1 "invalid" keypoint from the annotations)
    kps_valid_mask = (
        (src_kps[..., 0] >= 0) & (src_kps[..., 1] >= 0) &
        (trg_kps[..., 0] >= 0) & (trg_kps[..., 1] >= 0)
    )

    return {
        "src_img": src_img,
        "trg_img": trg_img,
        "src_kps": src_kps,
        "trg_kps": trg_kps,
        "src_nopad_size": src_nopad_size,
        "trg_nopad_size": trg_nopad_size,
        "kps_valid_mask": kps_valid_mask,
    }


def build_dataloaders(preprocess, dataset_size, num_workers, batch_size):
    pair_ann_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "PairAnnotation"
    layout_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "Layout"
    image_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "JPEGImages"
    pck_alpha = [0.05, 0.1, 0.2]

    train_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='trn', preprocess=preprocess)
    val_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='val', preprocess=preprocess)

    # pin_memory speeds up CPU->GPU transfers; persistent_workers avoids
    # respawning workers at every epoch (expensive on Windows)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    # Validation always uses batch_size=1 (variable keypoints + per-sample
    # nopad mask, as required by evaluate_one_epoch)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train, **loader_kwargs)
    val_dataloader = DataLoader(val_dataset, batch_size=1, **loader_kwargs)

    return train_dataloader, val_dataloader


def run_training(args, *, lr, unfreeze_layers, tau, effective_batch, dataset_size, save_path=None) -> float:
    """
    Fine-tune the model with validation at the end of each epoch.

    return: best PCK@0.10 (per point) on the validation set
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, preprocess = build_model_and_preprocess(args.model, args.checkpoint, device)
    backbone = get_backbone(model, args.model)

    trainable_params = unfreeze_last_layers(model, args.model, unfreeze_layers)
    n_trainable = sum(p.numel() for p in trainable_params)
    print(f"{args.model}: last {unfreeze_layers} layers unfrozen -> {n_trainable / 1e6:.1f}M trainable parameters")

    if args.compile:
        compile_backbone(backbone, args.model)
        print("torch.compile enabled: the first batches pay the compilation warmup")

    # Batch effettivo disaccoppiato dalla memoria: il forward usa il batch
    # reale massimo sostenibile dalla GPU e l'accumulo copre la differenza,
    # cosi' run con lo stesso effective_batch fanno gli stessi update per epoca
    max_real_batch = args.real_batch or MAX_REAL_BATCH[args.model] * (2 if args.amp else 1)
    batch_size = min(effective_batch, max_real_batch)
    grad_accum = math.ceil(effective_batch / batch_size)
    print(f"Effective batch {effective_batch} = {batch_size} per forward x {grad_accum} accumulation steps")

    train_dataloader, val_dataloader = build_dataloaders(preprocess, dataset_size, args.num_workers, batch_size)

    # fused=True esegue l'update di AdamW in un kernel CUDA unico
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=args.weight_decay, fused=(device.type == "cuda"))

    # Cosine decay over optimizer steps (scheduler.step() is called by
    # train_one_epoch after every optimizer.step()). T_max is defined on the
    # maximum number of epochs: with early stopping the cosine stays partial
    steps_per_epoch = math.ceil(len(train_dataloader) / grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs * steps_per_epoch,
        eta_min=lr * args.cosine_decay,
    )

    best_pck = 0.0
    epochs_no_improve = 0

    epoch_bar = tqdm(range(args.max_epochs), desc=f"{args.model} epochs", unit="epoch")
    for epoch in epoch_bar:
        backbone.train()
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_kwargs={"tau": tau},
            scheduler=scheduler,
            grad_accum_steps=grad_accum,
            max_grad_norm=args.max_grad_norm,
            amp=args.amp,
            epoch=epoch,
            log_wandb=True,
        )

        backbone.eval()
        metrics = evaluate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            method_name=f"{args.model} (epoch {epoch})",
            log_wandb=False,
        )

        pck = {name: metrics["point"][name]["mean"] for name in ("0.05", "0.10", "0.20")}

        epoch_bar.set_postfix(loss=f"{avg_loss:.3f}", pck10=f"{pck['0.10']:.2f}")

        wandb.log({
            "epoch": epoch,
            "train/avg_loss": avg_loss,
            "val/pck_0.05": pck["0.05"],
            "val/pck_0.10": pck["0.10"],
            "val/pck_0.20": pck["0.20"],
        })

        if pck["0.10"] > best_pck:
            best_pck = pck["0.10"]
            epochs_no_improve = 0

            if save_path is not None:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": args.model,
                    "epoch": epoch,
                    "val_pck_0.10": best_pck,
                    "config": {
                        "lr": lr,
                        "unfreeze_layers": unfreeze_layers,
                        "tau": tau,
                        "effective_batch": effective_batch,
                        "cosine_decay": args.cosine_decay,
                    },
                    "state_dict": backbone.state_dict(),
                }, save_path)
                print(f"New best (val PCK@0.10 = {best_pck:.2f}): checkpoint saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}: val PCK@0.10 stuck at {best_pck:.2f} for {args.patience} epochs")
                break

    if wandb.run is not None:
        # La summary di default e' l'ultimo wandb.log: best_run() e il ranking
        # dello sweep leggono questa chiave, quindi la sovrascriviamo col best
        wandb.run.summary["val/pck_0.10"] = best_pck
        wandb.run.summary["best/val_pck_0.10"] = best_pck
        wandb.run.summary["stopped_epoch"] = epoch

    return best_pck


def sweep_entry(args):
    """Single sweep run: hyperparameters come from wandb.config."""
    with wandb.init():
        # Summary = max storico per bayes/UI; best_run() pero' ignora
        # define_metric e legge la summary raw, che run_training sovrascrive
        # col best della run a fine training
        wandb.define_metric("val/pck_0.10", summary="max")
        config = wandb.config
        run_training(
            args,
            lr=config.lr,
            unfreeze_layers=args.unfreeze_layers,  # fisso da CLI per tutta la pipeline
            tau=config.tau,
            effective_batch=config.effective_batch,
            dataset_size='small',  # hyperparameter search on SPair small
        )


def run_sweep(args) -> dict:
    """
    Hyperparameter search with a W&B sweep on SPair small.

    return: hyperparameters of the best run (highest val/pck_0.10)
    """
    # Search space e pruning vivono in sweep_config.yaml; solo il nome dello
    # sweep dipende dal modello scelto da CLI
    with open(PROJECT_ROOT / "sweep_config.yaml", encoding="utf-8") as f:
        sweep_config = yaml.safe_load(f)
    sweep_config["name"] = f"{args.model}-finetune"

    print(f"Launching W&B sweep for {args.model}: {args.sweep_count} runs on SPair small")
    sweep_id = args.sweep_id or wandb.sweep(sweep_config, project=args.wandb_project)
    wandb.agent(sweep_id, function=lambda: sweep_entry(args), count=args.sweep_count, project=args.wandb_project)

    # L'agent passa run id e config a wandb.init tramite os.environ
    # (pyagent._run_job); la sua pulizia gira nel thread della run e se
    # hyperband uccide l'ultima run perde la corsa col wandb.init del
    # training finale, che riprenderebbe la run prunata (il server la ha
    # marcata "stop requested" e ucciderebbe anche il training finale)
    for var in (wandb.env.RUN_ID, wandb.env.SWEEP_ID, wandb.env.SWEEP_PARAM_PATH):
        os.environ.pop(var, None)

    # Fetch the best run of the finished sweep (ranked by the sweep metric)
    api = wandb.Api()
    sweep = api.sweep(f"{api.default_entity}/{args.wandb_project}/{sweep_id}")
    best_run = sweep.best_run()
    print(f"Sweep complete. Best run: {best_run.name} (val PCK@0.10 = {best_run.summary.get('best/val_pck_0.10', 'n/a')})")

    return {
        "lr": best_run.config["lr"],
        "tau": best_run.config["tau"],
        "effective_batch": best_run.config["effective_batch"],
    }


def main():
    args = parse_args()

    # La memory_efficient_attention di xformers non traccia sotto torch.compile
    # (bug di device propagation nel suo backward): disabilitandola, DinoV2
    # ripiega su F.scaled_dot_product_attention, equivalente per memoria e
    # velocita'. Va fatto prima che torch.hub importi i moduli dinov2.
    if args.compile:
        os.environ["XFORMERS_DISABLED"] = "1"

    if args.skip_sweep:
        # No search: use the CLI hyperparameters directly
        best_hparams = {
            "lr": args.lr,
            "tau": args.tau,
            "effective_batch": args.effective_batch,
        }
    else:
        # Hyperparameter search on SPair small before the actual training
        best_hparams = run_sweep(args)

    # Actual training on SPair large. unfreeze_layers e' fisso da CLI per
    # tutta la pipeline (il suo ottimo dipende dalla taglia del dataset,
    # quindi non viene cercato su small): per confrontare valori diversi si
    # rilancia lo script con un altro --unfreeze-layers
    config = {
        **best_hparams,
        "unfreeze_layers": args.unfreeze_layers,
        "cosine_decay": args.cosine_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "weight_decay": args.weight_decay,
        "amp": args.amp,
        "compile": args.compile,
        "dataset_size": "large",
    }

    # Log the selected hyperparameters to console and W&B before training
    print(f"Starting {args.model} fine-tuning with the selected hyperparameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    wandb.init(
        project=args.wandb_project,
        name=f"{args.model}-finetune-unfreeze{args.unfreeze_layers}",
        config=config,
    )
    wandb.define_metric("val/pck_0.10", summary="max")

    # Un checkpoint per valore di unfreeze: lanci diversi non si sovrascrivono
    save_path = PROJECT_ROOT / "checkpoints" / "finetune" / f"{args.model.lower()}_unfreeze{args.unfreeze_layers}_best.pth"
    best_pck = run_training(
        args,
        lr=best_hparams["lr"],
        unfreeze_layers=args.unfreeze_layers,
        tau=best_hparams["tau"],
        effective_batch=best_hparams["effective_batch"],
        dataset_size='large',
        save_path=save_path,
    )

    print(f"Training complete. Best val PCK@0.10: {best_pck:.2f} (checkpoint: {save_path.name})")
    wandb.finish()


if __name__ == '__main__':
    main()

from typing import Any, Callable

import torch
from tqdm import tqdm

from utils.loss import dense_infonce_loss


def train_one_epoch(
    *,
    model: Any,
    dataloader: Any,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable = dense_infonce_loss,
    loss_kwargs: dict | None = None,
    scheduler: Any | None = None,
    grad_accum_steps: int = 1,
    max_grad_norm: float | None = None,
    epoch: int | None = None,
    log_wandb: bool = False,
    log_every: int = 50,
) -> float:
    """
    Esegue un'epoca di training per semantic correspondence.

    model: wrapper con .forward(image) -> [B, C, Hf, Wf] (DinoV2/DinoV3/SAM)
    dataloader: batch con chiavi src_img, trg_img, src_kps, trg_kps
                e opzionalmente src_nopad_size, trg_nopad_size, kps_valid_mask
    loss_fn: firma compatibile con dense_infonce_loss
    loss_kwargs: extra kwargs inoltrati a loss_fn (es. {"tau": 0.05})
    scheduler: se presente, step() dopo ogni optimizer.step()
    grad_accum_steps: accumula i gradienti per N batch prima dello step
    max_grad_norm: se presente, clip della norma del gradiente

    return: loss media sull'epoca
    """
    loss_kwargs = loss_kwargs or {}

    desc = f"Training epoch {epoch}" if epoch is not None else "Training"

    running_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    for step, batch in enumerate(tqdm(dataloader, total=len(dataloader), desc=desc)):
        src_img = batch["src_img"]
        trg_img = batch["trg_img"]

        src_featuremap = model.forward(src_img)
        trg_featuremap = model.forward(trg_img)

        loss = loss_fn(
            src_feat=src_featuremap,
            trg_feat=trg_featuremap,
            src_kps=batch["src_kps"],
            trg_kps=batch["trg_kps"],
            src_image_size_pad=tuple(src_img.shape[-2:]),
            trg_image_size_pad=tuple(trg_img.shape[-2:]),
            src_nopad_size=batch.get("src_nopad_size"),
            trg_nopad_size=batch.get("trg_nopad_size"),
            kps_valid_mask=batch.get("kps_valid_mask"),
            **loss_kwargs,
        )

        (loss / grad_accum_steps).backward()

        if (step + 1) % grad_accum_steps == 0:
            if max_grad_norm is not None:
                params = [p for group in optimizer.param_groups for p in group["params"]]
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += loss.item()
        num_batches += 1

        if log_wandb and (step + 1) % log_every == 0:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                        **({"epoch": epoch} if epoch is not None else {}),
                    })
            except ImportError:
                pass

    # Step finale se l'epoca non e' multipla di grad_accum_steps
    if num_batches % grad_accum_steps != 0:
        if max_grad_norm is not None:
            params = [p for group in optimizer.param_groups for p in group["params"]]
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

    return running_loss / max(num_batches, 1)

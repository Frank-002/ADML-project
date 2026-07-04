import argparse
import sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import preprocess
from utils.featuremap import dense_correspondence
from utils.model_builder import build_model_and_preprocess
from utils.preprocess import PreProcess
from utils.results import compute_and_print_pck

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.SPairDataset import SPairDataset

def parse_args():
    parser = argparse.ArgumentParser()

    model = parser.add_subparsers(dest='model', required=True)
    dinov2 = model.add_parser("DINOV2")
    dinov2.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    dinov3 = model.add_parser("DINOV3")
    dinov3.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    sam = model.add_parser("SAM")
    sam.add_argument("--checkpoint", type=Path, required=False, help="path to checkpoint")

    return parser.parse_args()

def main():
    args = parse_args()

    pair_ann_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "PairAnnotation"
    layout_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "Layout"
    image_path = PROJECT_ROOT / "dataset" / "SPair-71k" / "JPEGImages"
    dataset_size = 'large'
    pck_alpha = [0.05, 0.1, 0.2]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, preprocess = build_model_and_preprocess(
        model_name=args.model,
        checkpoint=args.checkpoint,
        device=device,
        trainable=False
    )

    test_dataset = SPairDataset(pair_ann_path, layout_path, image_path, dataset_size, pck_alpha, datatype='test', preprocess=preprocess)
    test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1)

    results = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, total=len(test_dataloader), desc=f"Computing correspondences with {args.model}"):
            src_img = batch["src_img"]
            trg_img = batch["trg_img"]

            src_featuremap = model.forward(src_img)
            trg_featuremap = model.forward(trg_img)

            src_kps = batch["src_kps"]

            src_image_size_pad = src_img.shape[-2:]  # (H_pad, W_pad)
            trg_image_size_pad = trg_img.shape[-2:]  # (H_pad, W_pad)

            trg_nopad_size = batch["trg_nopad_size"]  # [B, 2] -> (H_nopad, W_nopad)

            pred_trg_kps = dense_correspondence(
                src_feat=src_featuremap,
                trg_feat=trg_featuremap,
                src_kps=src_kps,
                src_image_size_pad=src_image_size_pad,
                trg_image_size_pad=trg_image_size_pad,
                trg_size_nopad=(int(trg_nopad_size[0, 0]), int(trg_nopad_size[0, 1])),
            )

            out = {
                "category": batch["category"],
                "src_kps": src_kps.detach().cpu(),
                "pred_trg_kps": pred_trg_kps.detach().cpu(),
                "trg_kps": batch["trg_kps"],
                "pck_threshold": batch["pck_threshold"]

            }

            results.append(out)

    compute_and_print_pck(
        results,
        method_name=args.model,
        threshold_names=["0.05", "0.10", "0.20"],
        print_console=True,
        log_wandb=True,
    )




if __name__ == '__main__':
    main()
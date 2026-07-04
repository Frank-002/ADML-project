from pathlib import Path

import torch

from models.dinov2 import DinoV2
from models.dinov3 import DinoV3
from models.sam import SAM
from utils.preprocess import PreProcess


def build_model_and_preprocess(
        model_name: str,
        checkpoint: Path,
        device: torch.device,
        trainable: bool
):
    # In the sweep each run must start from a fresh model, so the
    # construction lives in a function rather than in main
    match model_name:
        case "DINOV2":
            model = DinoV2(device=device, trainable=trainable)
            preprocess = PreProcess(long_side_length=518, apply_norm=True)
        case "SAM":
            model = SAM(device=device, checkpoint=checkpoint, trainable=trainable)
            preprocess = PreProcess(long_side_length=1024, apply_norm=False)
        case "DINOV3":
            model = DinoV3(device=device, checkpoint=checkpoint, trainable=trainable)
            preprocess = PreProcess(long_side_length=512, apply_norm=True)
        case _:
            raise NotImplementedError

    return model, preprocess
from pathlib import Path

import torch

from models.dino_backbone import DinoBackbone
from models.sam import SAM
from utils.preprocess import PreProcess


def build_model_and_preprocess(
        model_name: str,
        checkpoint: Path,
        device: torch.device,
        trainable: bool
):
    match model_name:
        case "DINOV2":
            model = DinoBackbone(model_name=model_name, device=device, trainable=trainable)
            preprocess = PreProcess(long_side_length=518, apply_norm=True)
        case "SAM":
            model = SAM(device=device, checkpoint=checkpoint, trainable=trainable)
            # apply_norm=True sostituisce Sam.preprocess (stesse statistiche,
            # vedi models/sam.py); il pad avviene dopo la norm, come nella
            # pipeline SAM ufficiale
            preprocess = PreProcess(long_side_length=1024, apply_norm=True)
        case "DINOV3":
            model = DinoBackbone(model_name=model_name, device=device, checkpoint=checkpoint, trainable=trainable)
            preprocess = PreProcess(long_side_length=512, apply_norm=True)
        case _:
            raise NotImplementedError

    return model, preprocess

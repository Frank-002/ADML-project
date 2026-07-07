from pathlib import Path

import torch

from models.dino_backbone import DinoBackbone
from models.sam import SAM
from utils.preprocess import PreProcess


def compile_backbone(backbone, model_name, mode=None) -> None:
    """
    torch.compile del percorso pesante del forward, in-place: niente wrapper
    OptimizedModule sul modulo, cosi' le chiavi dello state_dict (e quindi i
    checkpoint letti da eval.py) restano identiche.
    """
    if model_name in ("DINOV2", "DINOV3"):
        backbone.forward_features = torch.compile(backbone.forward_features, mode=mode)
    elif model_name == "SAM":
        backbone.image_encoder.compile()
    else:
        raise NotImplementedError


def build_model_and_preprocess(
        model_name: str,
        checkpoint: Path,
        device: torch.device,
        trainable: bool
):
    match model_name:
        case "DINOV2":
            model = DinoBackbone(model_name=model_name, device=device, checkpoint=checkpoint, trainable=trainable)
            preprocess = PreProcess(long_side_length=518, apply_norm=True)
        case "SAM":
            model = SAM(device=device, checkpoint=checkpoint, trainable=trainable)
            # apply_norm=True sostituisce Sam.preprocess (stesse statistiche,
            # vedere models/sam.py); il pad avviene dopo la norm, come nella
            # pipeline SAM ufficiale
            preprocess = PreProcess(long_side_length=1024, apply_norm=True)
        case "DINOV3":
            model = DinoBackbone(model_name=model_name, device=device, checkpoint=checkpoint, trainable=trainable)
            preprocess = PreProcess(long_side_length=768, apply_norm=True)
        case _:
            raise NotImplementedError

    return model, preprocess

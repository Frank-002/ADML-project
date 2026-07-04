from pathlib import Path

import torch

from models.DinoBackbone import DinoBackbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DinoV3(DinoBackbone):
    def __init__(
            self,
            *,
            device: torch.device,
            checkpoint: Path,
            trainable: bool = False
    ):
        model = torch.hub.load(
            str(PROJECT_ROOT / "dinov3-git"),
            'dinov3_vitb16',
            source='local',
            weights=str(checkpoint))
        super().__init__(
            model,
            device,
            model.patch_size,
            trainable
        )

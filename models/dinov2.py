import torch

from models.DinoBackbone import DinoBackbone


class DinoV2(DinoBackbone):
    def __init__(
            self,
            *,
            device: torch.device,
            trainable: bool = False
    ):
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        super().__init__(
            model,
            device,
            model.patch_size,
            trainable
        )

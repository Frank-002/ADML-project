from pathlib import Path

import torch
from segment_anything import sam_model_registry

from models.backbone import Backbone


class SAM(Backbone):
    def __init__(
            self,
            *,
            device: torch.device,
            checkpoint: Path,
            trainable: bool = False
    ):
        model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        patch_size = int(model.image_encoder.patch_embed.proj.kernel_size[0])
        super().__init__(
            model,
            device,
            patch_size,
            trainable
        )

    def forward(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        # L'input arriva gia' normalizzato e paddato a 1024 da PreProcess
        # (apply_norm=True: le pixel_mean/std di Sam.preprocess sono le
        # statistiche ImageNet in scala 0-255), quindi Sam.preprocess non va
        # richiamato: l'immagine entra diretta nell'image_encoder
        image = image.to(self.device)
        return self.model.image_encoder(image)

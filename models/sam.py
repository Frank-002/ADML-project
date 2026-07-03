from pathlib import Path

import torch
from segment_anything import sam_model_registry, SamPredictor


class SAM():
    def __init__(
            self,
            *,
            device: torch.device,
            checkpoint: Path,
            trainable: bool = False
    ):
        self.device = device
        self.trainable = trainable
        model = sam_model_registry["vit_b"](checkpoint=checkpoint)
        if trainable:
            model.to(self.device).train()
        else:
            model.to(self.device).eval()

        self.model = SamPredictor(model)

        self.patch_size = int(self.model.model.image_encoder.patch_embed.proj.kernel_size[0])

    def forward(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        if self.trainable:
            # Replica SamPredictor.set_torch_image (preprocess + image_encoder)
            # senza il suo @torch.no_grad(), che bloccherebbe il fine-tuning
            image = self.model.model.preprocess(image.to(self.device))
            features = self.model.model.image_encoder(image)
        else:
            self.model.set_torch_image(image.to(self.device), (1024, 1024))
            features = self.model.get_image_embedding()
        return features
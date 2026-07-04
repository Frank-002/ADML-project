from pathlib import Path

import torch

from models.backbone import Backbone

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DinoBackbone(Backbone):
    def __init__(
            self,
            *,
            model_name: str,
            device: torch.device,
            checkpoint: Path | None = None,
            trainable: bool = False,
    ):
        match model_name:
            case "DINOV2":
                model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            case "DINOV3":
                # I pesi base di DinoV3 sono gated: vanno scaricati a parte
                # e passati via checkpoint
                model = torch.hub.load(
                    str(PROJECT_ROOT / "dinov3-git"),
                    'dinov3_vitb16',
                    source='local',
                    weights=str(checkpoint))
            case _:
                raise NotImplementedError(model_name)
        # TODO: caricamento dei checkpoint fine-tunati salvati da train.py
        #  ({"state_dict": ...}): andra' qui, comune a DINOV2 e DINOV3
        super().__init__(
            model,
            device,
            model.patch_size,
            trainable
        )

    def forward(
            self,
            image: torch.Tensor,
    ) -> torch.Tensor:
        image = image.to(self.device)

        features = self.model.forward_features(image)
        patch_features = features["x_norm_patchtokens"]  # [B, N, C]

        B, N, C = patch_features.shape
        _, _, H_img, W_img = image.shape

        patch_h = patch_w = self.patch_size

        H_feat = H_img // patch_h
        W_feat = W_img // patch_w

        # [B, N, C] -> [B, H_feat, W_feat, C]
        patch_features = patch_features.view(B, H_feat, W_feat, C)

        # [B, H_feat, W_feat, C] -> [B, C, H_feat, W_feat]
        patch_features = patch_features.permute(0, 3, 1, 2).contiguous()

        return patch_features

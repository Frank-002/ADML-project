from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DinoV3:
    def __init__(
            self,
            *,
            device: torch.device,
            checkpoint: Path,
            trainable: bool = False
    ):
        self.device = device

        model = torch.hub.load(
            str(PROJECT_ROOT / "dinov3-git"),
            'dinov3_vitb16',
            source='local',
            weights=str(checkpoint))

        if trainable:
            self.model = model.to(self.device).train()
        else:
            self.model = model.to(self.device).eval()

        self.patch_size = self.model.patch_size

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
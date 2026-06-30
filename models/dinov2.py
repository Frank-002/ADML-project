import torch
from torchvision import transforms


class DinoV2:
    def __init__(
            self,
            *,
            device: torch.device,
            trainable: bool = False
    ):
        self.device = device
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
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

        if isinstance(self.patch_size, tuple):
            patch_h, patch_w = self.patch_size
        else:
            patch_h = patch_w = self.patch_size

        H_feat = H_img // patch_h
        W_feat = W_img // patch_w

        if N != H_feat * W_feat:
            raise ValueError(
                f"Numero di patch non coerente: N={N}, "
                f"H_feat*W_feat={H_feat * W_feat}, "
                f"image.shape={image.shape}, "
                f"patch_size={(patch_h, patch_w)}"
            )

        # [B, N, C] -> [B, H_feat, W_feat, C]
        patch_features = patch_features.view(B, H_feat, W_feat, C)

        # [B, H_feat, W_feat, C] -> [B, C, H_feat, W_feat]
        patch_features = patch_features.permute(0, 3, 1, 2).contiguous()

        return patch_features

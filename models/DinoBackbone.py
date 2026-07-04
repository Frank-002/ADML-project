import torch

from models.Backbone import Backbone


class DinoBackbone(Backbone):
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

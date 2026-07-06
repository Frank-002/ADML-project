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
                if checkpoint is not None:
                    # Checkpoint fine-tunati salvati da train.py: le chiavi di
                    # state_dict sono quelle del modello hub (train.py compila
                    # in-place proprio per non alterarle)
                    state = torch.load(checkpoint, map_location="cpu")
                    model.load_state_dict(state["state_dict"])
            case "DINOV3":
                # checkpoint accetta sia i pesi base (gated, scaricati a
                # parte) sia un checkpoint fine-tunato di train.py: i due
                # formati si distinguono dalla chiave "state_dict"
                state = torch.load(checkpoint, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    # Checkpoint di train.py: contiene lo state_dict completo
                    # del backbone (chiavi del modello hub), i pesi base non
                    # servono
                    model = torch.hub.load(
                        str(PROJECT_ROOT / "dinov3-git"),
                        'dinov3_vitb16',
                        source='local',
                        pretrained=False)
                    model.load_state_dict(state["state_dict"])
                else:
                    model = torch.hub.load(
                        str(PROJECT_ROOT / "dinov3-git"),
                        'dinov3_vitb16',
                        source='local',
                        weights=str(checkpoint))
            case _:
                raise NotImplementedError(model_name)
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

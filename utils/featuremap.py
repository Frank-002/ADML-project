import math

import torch
import torch.nn.functional as F

def normalize_features(feat: torch.Tensor) -> torch.Tensor:
    return F.normalize(feat, p=2, dim=1)

def sample_features_at_keypoints(
    feat: torch.Tensor,
    kps: torch.Tensor,
    image_size: tuple[int, int],
) -> torch.Tensor:
    """
    feat: [B, C, Hf, Wf]
    kps: [B, K, 2] in coordinate immagine riscalata, formato (x, y)
    image_size: (H_img, W_img)

    return: [B, K, C]
    """

    B, C, Hf, Wf = feat.shape
    H_img, W_img = image_size

    kps = kps.to(feat.device).float()

    x = kps[..., 0]
    y = kps[..., 1]

    # Convenzione area / centro-patch (align_corners=False):
    # il pixel immagine al centro della cella feature j si mappa sul centro
    # della cella, coerentemente con la semantica dei patch del ViT.
    x_norm = 2.0 * (x + 0.5) / W_img - 1.0
    y_norm = 2.0 * (y + 0.5) / H_img - 1.0

    grid = torch.stack([x_norm, y_norm], dim=-1)  # [B, K, 2]
    grid = grid.unsqueeze(2)                      # [B, K, 1, 2]

    sampled = F.grid_sample(
        feat,
        grid,
        mode="bilinear",
        align_corners=False,
    )  # [B, C, K, 1]

    sampled = sampled.squeeze(-1).permute(0, 2, 1).contiguous()  # [B, K, C]
    return sampled

def make_valid_feature_mask(
    feat: torch.Tensor,
    image_size_pad: tuple[int, int],
    size_nopad: tuple[int, int],
) -> torch.Tensor:
    """
    feat: [B, C, Hf, Wf]
    image_size_pad: (H_pad, W_pad)
    size_nopad: (H_no_pad, W_no_pad)

    return: [B, 1, Hf, Wf]
    """

    B, _, Hf, Wf = feat.shape
    H_pad, W_pad = image_size_pad
    H_no_pad, W_no_pad = size_nopad

    H_keep = math.ceil(Hf * H_no_pad / H_pad)
    W_keep = math.ceil(Wf * W_no_pad / W_pad)

    mask = torch.zeros(
        B, 1, Hf, Wf,
        dtype=torch.bool,
        device=feat.device,
    )

    mask[:, :, :H_keep, :W_keep] = True

    return mask

def dense_correspondence(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_kps: torch.Tensor,
    src_image_size_pad: tuple[int, int],
    trg_image_size_pad: tuple[int, int],
    trg_size_nopad: tuple[int, int] | None = None,
) -> torch.Tensor:
    """
    src_feat: [B, C, Hs, Ws]
    trg_feat: [B, C, Ht, Wt]
    src_kps: [B, K, 2] coordinate (x, y) nello spazio della padded image

    return: [B, K, 2] coordinate predette sul target padded
    """

    src_feat = normalize_features(src_feat)
    trg_feat = normalize_features(trg_feat)

    B, C, Ht, Wt = trg_feat.shape

    src_desc = sample_features_at_keypoints(
        feat=src_feat,
        kps=src_kps,
        image_size=src_image_size_pad,
    )  # [B, K, C]

    src_desc = F.normalize(src_desc, p=2, dim=-1)

    trg_flat = trg_feat.flatten(2)  # [B, C, Ht * Wt]

    sim = torch.einsum("bkc,bcn->bkn", src_desc, trg_flat)  # [B, K, Ht * Wt]

    # Escludi il padding del target dall'argmax
    if trg_size_nopad is not None:
        trg_valid_mask = make_valid_feature_mask(
            feat=trg_feat,
            image_size_pad=trg_image_size_pad,
            size_nopad=trg_size_nopad,
        )  # [B, 1, Ht, Wt]

        trg_valid_mask = trg_valid_mask.flatten(2)  # [B, 1, Ht * Wt]
        sim = sim.masked_fill(~trg_valid_mask, float("-inf"))

    nn_idx = sim.argmax(dim=-1)  # [B, K]

    y_feat = nn_idx // Wt
    x_feat = nn_idx % Wt

    H_trg_pad, W_trg_pad = trg_image_size_pad

    # Inversa della convenzione area / centro-patch (align_corners=False):
    # la cella feature j corrisponde al centro patch ((j + 0.5) * stride - 0.5).
    x_img = (x_feat.float() + 0.5) * W_trg_pad / Wt - 0.5
    y_img = (y_feat.float() + 0.5) * H_trg_pad / Ht - 0.5

    pred_trg_kps = torch.stack([x_img, y_img], dim=-1)  # [B, K, 2]

    return pred_trg_kps
import torch
import torch.nn.functional as F

from utils.featuremap import (
    keypoints_to_feature_indices,
    make_valid_feature_mask,
    normalize_features,
    sample_features_at_keypoints,
)


def _one_way_infonce(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    src_image_size_pad: tuple[int, int],
    trg_image_size_pad: tuple[int, int],
    tau: float,
    trg_nopad_size: torch.Tensor | None,
    kps_valid_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    InfoNCE src -> trg: per ogni keypoint sorgente, cross-entropy sulla
    similarita' contro tutti i patch del target; positivo = patch GT.

    src_feat / trg_feat: [B, C, H, W] gia' L2-normalizzate sul canale
    src_kps / trg_kps: [B, K, 2] coordinate (x, y) nello spazio padded
    trg_nopad_size: [B, 2] per mascherare il padding del target, o None
    kps_valid_mask: [B, K] bool, o None se tutti i keypoint sono validi
    """
    B, _, Ht, Wt = trg_feat.shape
    N = Ht * Wt

    src_desc = sample_features_at_keypoints(
        feat=src_feat,
        kps=src_kps,
        image_size=src_image_size_pad,
    )  # [B, K, C]
    src_desc = F.normalize(src_desc, p=2, dim=-1)

    trg_flat = trg_feat.flatten(2)  # [B, C, N]
    logits = torch.einsum("bkc,bcn->bkn", src_desc, trg_flat) / tau  # [B, K, N]

    # Escludi il padding del target dai negativi (il GT non cade mai nel
    # padding, quindi il positivo resta sempre disponibile)
    if trg_nopad_size is not None:
        trg_valid = make_valid_feature_mask(
            feat=trg_feat,
            image_size_pad=trg_image_size_pad,
            size_nopad=trg_nopad_size,
        )  # [B, 1, Ht, Wt]
        logits = logits.masked_fill(~trg_valid.flatten(2), float("-inf"))

    target_idx = keypoints_to_feature_indices(
        kps=trg_kps.to(logits.device),
        feat_size=(Ht, Wt),
        image_size_pad=trg_image_size_pad,
    )  # [B, K]

    K = target_idx.shape[1]
    loss = F.cross_entropy(
        logits.reshape(B * K, N),
        target_idx.reshape(B * K),
        reduction="none",
    ).view(B, K)

    if kps_valid_mask is not None:
        kps_valid_mask = kps_valid_mask.to(loss.device)
        loss = (loss * kps_valid_mask).sum() / kps_valid_mask.sum().clamp(min=1)
    else:
        loss = loss.mean()

    return loss


def dense_infonce_loss(
    src_feat: torch.Tensor,
    trg_feat: torch.Tensor,
    src_kps: torch.Tensor,
    trg_kps: torch.Tensor,
    src_image_size_pad: tuple[int, int],
    trg_image_size_pad: tuple[int, int],
    tau: float = 0.05,
    src_nopad_size: torch.Tensor | None = None,
    trg_nopad_size: torch.Tensor | None = None,
    kps_valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    InfoNCE densa bidirezionale per semantic correspondence.

    src_feat / trg_feat: [B, C, H, W] feature map dei due rami
    src_kps / trg_kps: [B, K, 2] keypoint corrispondenti (x, y), spazio padded
    *_image_size_pad: (H_pad, W_pad) delle immagini in input al modello
    *_nopad_size: [B, 2] -> (H_nopad, W_nopad) per mascherare il padding
    kps_valid_mask: [B, K] bool per keypoint padded nel collate (None = tutti validi)
    tau: temperatura dei logit

    return: scalare, media delle due direzioni src->trg e trg->src
    """
    src_feat = normalize_features(src_feat)
    trg_feat = normalize_features(trg_feat)

    loss_st = _one_way_infonce(
        src_feat=src_feat,
        trg_feat=trg_feat,
        src_kps=src_kps,
        trg_kps=trg_kps,
        src_image_size_pad=src_image_size_pad,
        trg_image_size_pad=trg_image_size_pad,
        tau=tau,
        trg_nopad_size=trg_nopad_size,
        kps_valid_mask=kps_valid_mask,
    )

    loss_ts = _one_way_infonce(
        src_feat=trg_feat,
        trg_feat=src_feat,
        src_kps=trg_kps,
        trg_kps=src_kps,
        src_image_size_pad=trg_image_size_pad,
        trg_image_size_pad=src_image_size_pad,
        tau=tau,
        trg_nopad_size=src_nopad_size,
        kps_valid_mask=kps_valid_mask,
    )

    return 0.5 * (loss_st + loss_ts)

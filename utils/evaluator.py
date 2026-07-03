from typing import Any, Callable

import torch
from tqdm import tqdm

from utils.featuremap import dense_correspondence
from utils.results import compute_and_print_pck


@torch.no_grad()
def evaluate_one_epoch(
    *,
    model: Any,
    dataloader: Any,
    method_name: str = "model",
    correspondence_fn: Callable = dense_correspondence,
    threshold_names: list[str] | None = None,
    print_console: bool = True,
    log_wandb: bool = False,
    return_results: bool = False,
):
    """
    Esegue un'epoca di valutazione (PCK) per semantic correspondence.

    model: wrapper con .forward(image) -> [B, C, Hf, Wf] (DinoV2/DinoV3/SAM)
    dataloader: batch con chiavi src_img, trg_img, src_kps, trg_kps,
                trg_nopad_size, pck_threshold, category (batch_size=1)
    correspondence_fn: firma compatibile con dense_correspondence
    threshold_names: etichette delle soglie PCK (default 0.05/0.10/0.20)

    return: dict di metriche da compute_and_print_pck;
            se return_results=True, la tupla (metrics, results)
    """
    results = []

    for batch in tqdm(dataloader, total=len(dataloader), desc=f"Evaluating {method_name}"):
        src_img = batch["src_img"]
        trg_img = batch["trg_img"]

        if src_img.shape[0] != 1:
            raise NotImplementedError(
                "evaluate_one_epoch supporta solo batch_size=1 "
                "(keypoint variabili per coppia e nopad mask per-sample)"
            )

        src_featuremap = model.forward(src_img)
        trg_featuremap = model.forward(trg_img)

        src_kps = batch["src_kps"]

        trg_nopad_size = batch.get("trg_nopad_size")  # [B, 2] -> (H_nopad, W_nopad)
        if trg_nopad_size is not None:
            trg_size_nopad = (int(trg_nopad_size[0, 0]), int(trg_nopad_size[0, 1]))
        else:
            trg_size_nopad = None

        pred_trg_kps = correspondence_fn(
            src_feat=src_featuremap,
            trg_feat=trg_featuremap,
            src_kps=src_kps,
            src_image_size_pad=tuple(src_img.shape[-2:]),
            trg_image_size_pad=tuple(trg_img.shape[-2:]),
            trg_size_nopad=trg_size_nopad,
        )

        results.append({
            "category": batch["category"],
            "src_kps": src_kps.detach().cpu(),
            "pred_trg_kps": pred_trg_kps.detach().cpu(),
            "trg_kps": batch["trg_kps"],
            "pck_threshold": batch["pck_threshold"],
        })

    metrics = compute_and_print_pck(
        results,
        method_name=method_name,
        threshold_names=threshold_names or ["0.05", "0.10", "0.20"],
        print_console=print_console,
        log_wandb=log_wandb,
    )

    if return_results:
        return metrics, results

    return metrics

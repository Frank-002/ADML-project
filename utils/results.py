from collections import defaultdict

import torch


def compute_and_print_pck(
    results: list[dict],
    method_name: str = "model",
    threshold_names: list[str] | None = None,
    categories_order: list[str] | None = None,
    invalid_value: float = -1.0,
    print_console: bool = True,
    log_wandb: bool = False,
    wandb_project: str = "ADML-project",
    wandb_run_name: str | None = None,
):
    """
    Compute and (optionally) print / log to Weights & Biases:
        - PCK per point results
        - PCK per image results

    Each entry of `results` must contain:
        out = {
            "category": category,
            "trg_kps": trg_kps.detach().cpu(),
            "pred_trg_kps": pred_trg_kps.detach().cpu(),
            "pck_threshold": batch["pck_threshold"],
        }

    `pck_threshold` holds the already-computed absolute thresholds, e.g.:
        [thr_0.05, thr_0.10, thr_0.20]

    Args:
        print_console: if True, print the tables to the console.
        log_wandb: if True, log the metrics to Weights & Biases.
        wandb_project: W&B project name (used only if no run is active).
        wandb_run_name: W&B run name (defaults to method_name).

    Returns:
        A dict with the computed metrics ("point" and "image"), handy for later use.
    """

    if categories_order is None:
        categories_order = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "dog", "horse",
            "motorbike", "person", "pottedplant", "sheep",
            "train", "tvmonitor",
        ]

    # Infer how many thresholds are present from the first result.
    first_threshold = results[0]["pck_threshold"]
    if torch.is_tensor(first_threshold):
        first_threshold = first_threshold.detach().cpu()
    else:
        first_threshold = torch.tensor(first_threshold)

    if first_threshold.ndim == 0:
        num_thresholds = 1
    elif first_threshold.ndim == 1:
        num_thresholds = first_threshold.numel()
    else:
        num_thresholds = first_threshold.shape[-1]

    if threshold_names is None:
        threshold_names = ["0.05", "0.10", "0.20"][:num_thresholds]

    # PCK per point accumulators (correct / valid keypoint counts per category).
    point_correct = [defaultdict(int) for _ in range(num_thresholds)]
    point_valid = [defaultdict(int) for _ in range(num_thresholds)]

    # PCK per image accumulator: store each image's PCK, then average per category.
    image_scores = [defaultdict(list) for _ in range(num_thresholds)]

    for out in results:
        pred = out["pred_trg_kps"].detach().cpu().float()
        gt = out["trg_kps"].detach().cpu().float()

        if pred.ndim == 2:
            pred = pred.unsqueeze(0)  # [1, K, 2]

        if gt.ndim == 2:
            gt = gt.unsqueeze(0)      # [1, K, 2]

        batch_size, num_kps, _ = pred.shape

        thresholds = out["pck_threshold"]
        if torch.is_tensor(thresholds):
            thresholds = thresholds.detach().cpu().float()
        else:
            thresholds = torch.tensor(thresholds).float()

        # Bring thresholds to shape [batch_size, num_thresholds].
        if thresholds.ndim == 0:
            thresholds = thresholds.view(1, 1).repeat(batch_size, 1)
        elif thresholds.ndim == 1:
            thresholds = thresholds.view(1, -1).repeat(batch_size, 1)
        elif thresholds.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported pck_threshold shape: {thresholds.shape}")

        category = out.get("category", "unknown")

        if torch.is_tensor(category):
            category = category.detach().cpu().tolist()

        if isinstance(category, (list, tuple)):
            categories = [str(c) for c in category]
        else:
            categories = [str(category)] * batch_size

        distances = torch.norm(pred - gt, dim=-1)  # [batch_size, num_kps]

        # A keypoint is valid only if its ground-truth coordinates are set.
        valid = (
            (gt[..., 0] > invalid_value) &
            (gt[..., 1] > invalid_value)
        )  # [batch_size, num_kps]

        for t_idx in range(num_thresholds):
            abs_threshold = thresholds[:, t_idx].view(batch_size, 1)  # [batch_size, 1]
            correct = (distances <= abs_threshold) & valid  # [batch_size, num_kps]

            for b in range(batch_size):
                cat = categories[b]

                n_valid = int(valid[b].sum().item())
                n_correct = int(correct[b].sum().item())

                # PCK per point: accumulate raw counts.
                point_valid[t_idx][cat] += n_valid
                point_correct[t_idx][cat] += n_correct

                # PCK per image: store this image's score.
                if n_valid > 0:
                    image_pck = 100.0 * n_correct / n_valid
                    image_scores[t_idx][cat].append(image_pck)

    def format_value(value):
        """Format a PCK value, or a dash when the category has no samples."""
        if value is None:
            return "-"
        return f"{value:.1f}"

    def compute_point_rows():
        """One row per threshold: (threshold_name, per-category values, mean)."""
        rows = []

        for t_idx in range(num_thresholds):
            values = []

            for cat in categories_order:
                valid_count = point_valid[t_idx][cat]
                correct_count = point_correct[t_idx][cat]

                if valid_count == 0:
                    values.append(None)
                else:
                    values.append(100.0 * correct_count / valid_count)

            present_values = [v for v in values if v is not None]
            mean_value = sum(present_values) / max(len(present_values), 1)

            rows.append((threshold_names[t_idx], values, mean_value))

        return rows

    def compute_image_rows():
        """One row per threshold: (threshold_name, per-category values, mean)."""
        rows = []

        for t_idx in range(num_thresholds):
            values = []

            for cat in categories_order:
                scores = image_scores[t_idx][cat]

                if len(scores) == 0:
                    values.append(None)
                else:
                    values.append(sum(scores) / len(scores))

            present_values = [v for v in values if v is not None]
            mean_value = sum(present_values) / max(len(present_values), 1)

            rows.append((threshold_names[t_idx], values, mean_value))

        return rows

    def print_section(title: str, rows):
        """
        Print a section with categories as rows and thresholds as columns,
        which stays readable even with full category names.
        """
        name_width = max([len(c) for c in categories_order] + [len("Category")]) + 2
        col_width = 10

        threshold_labels = [f"a={alpha}" for alpha, _, _ in rows]

        header = f"{'Category':<{name_width}}" + "".join(
            f"{label:>{col_width}}" for label in threshold_labels
        )
        double_line = "=" * len(header)
        single_line = "-" * len(header)

        print()
        print(double_line)
        print(f"{title} - {method_name}")
        print(double_line)
        print(header)
        print(single_line)

        # One row per category, one column per threshold.
        for cat_idx, cat in enumerate(categories_order):
            row = f"{cat:<{name_width}}"
            for _, values, _ in rows:
                row += f"{format_value(values[cat_idx]):>{col_width}}"
            print(row)

        print(single_line)

        # Mean row across all categories.
        mean_row = f"{'Mean':<{name_width}}"
        for _, _, mean_value in rows:
            mean_row += f"{format_value(mean_value):>{col_width}}"
        print(mean_row)
        print(double_line)

    def build_metrics(point_rows, image_rows):
        """Build a nested dict holding every computed metric."""
        metrics = {"point": {}, "image": {}}

        for section, rows in (("point", point_rows), ("image", image_rows)):
            for alpha, values, mean_value in rows:
                per_category = {
                    cat: value
                    for cat, value in zip(categories_order, values)
                    if value is not None
                }
                metrics[section][alpha] = {
                    "per_category": per_category,
                    "mean": mean_value,
                }

        return metrics

    def log_to_wandb(point_rows, image_rows):
        try:
            import wandb
        except ImportError:
            print("[wandb] package not installed: logging skipped.")
            return

        run = wandb.run
        if run is None:
            run = wandb.init(
                project=wandb_project,
                name=wandb_run_name or method_name,
            )

        log_dict = {}
        columns = ["alpha", *categories_order, "Mean"]
        tables = {}

        for section, rows in (("point", point_rows), ("image", image_rows)):
            table = wandb.Table(columns=columns)

            for alpha, values, mean_value in rows:
                # Scalar metrics (useful for charts / cross-run comparisons).
                for cat, value in zip(categories_order, values):
                    if value is not None:
                        log_dict[f"pck_{section}/alpha_{alpha}/{cat}"] = value
                log_dict[f"pck_{section}/alpha_{alpha}/mean"] = mean_value

                # Summary table row.
                table_row = [
                    alpha,
                    *[
                        round(value, 1) if value is not None else None
                        for value in values
                    ],
                    round(mean_value, 1),
                ]
                table.add_data(*table_row)

            tables[f"pck_{section}_table"] = table

        run.log({**log_dict, **tables})
        print(f"[wandb] metrics logged to run '{run.name}'.")

    point_rows = compute_point_rows()
    image_rows = compute_image_rows()

    if print_console:
        print_section("PCK per point results", point_rows)
        print_section("PCK per image results", image_rows)

    if log_wandb:
        log_to_wandb(point_rows, image_rows)

    return build_metrics(point_rows, image_rows)

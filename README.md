# Semantic Correspondence with Vision Foundation Models

Fine-tuning and benchmarking of vision foundation backbones (**DINOv2**, **DINOv3**, **SAM**) on **dense semantic correspondence** over the [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/) benchmark.

> Solo project by **Anonymous** — Advanced Deep Machine Learning (ADML) course.

## Overview

Given a pair of images of the same object category, the task is to match keypoints from the source image to the target image (*semantic correspondence*). Correspondences are computed densely in feature space: each source keypoint is matched to the target patch with the most similar backbone feature.

The project compares three foundation backbones, both **off-the-shelf** (frozen pretrained features) and after a **partial fine-tuning**:

| Backbone | Variant | Feature source |
|---|---|---|
| DINOv2 | ViT-B/14 | patch tokens (`x_norm_patchtokens`) |
| DINOv3 | ViT-B/16 | patch tokens (`x_norm_patchtokens`) |
| SAM | ViT-B | image encoder feature map |

Fine-tuning uses a **symmetric InfoNCE contrastive loss** on ground-truth keypoint correspondences (averaged over the src→trg and trg→src directions), unfreezing only the last *N* transformer blocks plus the final norm/neck (`--unfreeze-layers`) while keeping the rest of the backbone frozen.

## Results

Evaluation on the SPair-71k **test split (large)**, reporting per-point PCK@α (Percentage of Correct Keypoints, threshold relative to the target bounding box), averaged over the 18 object categories.

| Model | PCK@0.05 | PCK@0.10 | PCK@0.20 |
|---|---|---|---|
| DINOv2 ViT-B/14 (pretrained) | TBD | TBD | TBD |
| DINOv2 ViT-B/14 (fine-tuned) | TBD | TBD | TBD |
| DINOv3 ViT-B/16 (pretrained) | TBD | TBD | TBD |
| DINOv3 ViT-B/16 (fine-tuned) | TBD | TBD | TBD |
| SAM ViT-B (pretrained) | TBD | TBD | TBD |
| SAM ViT-B (fine-tuned) | TBD | TBD | TBD |

> **Hyperparameter search.** All fine-tuning hyperparameters (learning rate, InfoNCE temperature τ, effective batch size) were found with a [Weights & Biases](https://wandb.ai) **Bayesian sweep** with Hyperband early termination, run on the SPair-71k *small* split before the final training on *large* (see [`sweep_config.yaml`](sweep_config.yaml)). Training curves, sweep results and full metrics are logged to the `ADML-project` W&B project.

## Setup

### Requirements

- Python 3.12 with a CUDA-enabled GPU (defaults are sized for ~12 GB of VRAM; use `--real-batch` on smaller cards)
- PyTorch 2.10 with CUDA 13.0 (see `requirements.txt`)

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130
```

The extra index is needed for the CUDA builds of `torch` / `torchvision` (`+cu130`), which are not published on PyPI.

`torch.compile` is used to speed up the backbone forward and requires Triton — on Windows: `pip install triton-windows`. It is on by default in evaluation (`--no-compile` to disable) and opt-in during training (`--compile`).

### Dataset

Download [SPair-71k](http://cvlab.postech.ac.kr/research/SPair-71k/) and extract it under `dataset/`:

```
dataset/SPair-71k/
├── JPEGImages/
├── PairAnnotation/
└── Layout/          # both "large" and "small" splits are used
```

### Model weights

- **DINOv2** — downloaded automatically from `torch.hub` (`dinov2_vitb14`).
- **DINOv3** — two steps, since neither the code nor the weights can be redistributed here:
  1. Clone the official repo into the project root (it is used as the local `torch.hub` source):
     ```bash
     git clone https://github.com/facebookresearch/dinov3.git dinov3-git
     ```
  2. The pretrained weights are gated: request access from [Meta](https://github.com/facebookresearch/dinov3), download `dinov3_vitb16`, and pass the file via `--checkpoint`.
- **SAM** — download the ViT-B checkpoint (`sam_vit_b_01ec64.pth`) from [segment-anything](https://github.com/facebookresearch/segment-anything#model-checkpoints) and pass it via `--checkpoint`.

### Weights & Biases

Training runs the hyperparameter sweep and logs all metrics to [W&B](https://wandb.ai) (evaluation also logs its final PCK tables). You need your own free W&B account:

```bash
wandb login
```

Runs are logged under your account to the project set by `--wandb-project` (default: `ADML-project`).

## Usage

Run `python eval.py --help` / `python train.py --help` for the full list of options.

### Evaluation

Evaluates PCK@{0.05, 0.10, 0.20} on the SPair-71k test split:

```bash
# Pretrained backbones
python eval.py DINOV2
python eval.py DINOV3 --checkpoint path/to/dinov3_vitb16_weights.pth
python eval.py SAM    --checkpoint path/to/sam_vit_b_01ec64.pth

# Fine-tuned checkpoint saved by train.py
python eval.py DINOV2 --checkpoint checkpoints/finetune/dinov2_unfreeze4_best.pth
python eval.py DINOV3 --checkpoint checkpoints/finetune/dinov3_unfreeze4_best.pth
```

> **Note.** For DINOv3, `--checkpoint` accepts either the base pretrained weights (official format) or a fine-tuned checkpoint saved by `train.py` — the format is detected automatically. For SAM, `--checkpoint` always refers to the base pretrained weights. Loading a fine-tuned SAM checkpoint is not supported yet.

### Training

Training runs in two stages, driven by a single command:

1. **Hyperparameter search** — a W&B Bayesian sweep (`--sweep-count` runs, default 12) on the SPair-71k *small* split, searching learning rate, τ and effective batch size, with Hyperband pruning of weak runs.
2. **Final training** — on the SPair-71k *large* split with the best hyperparameters from the sweep. bf16 mixed precision, cosine LR decay, and early stopping on validation PCK@0.10.

```bash
python train.py DINOV2 --unfreeze-layers 4

# DINOv3 and SAM need their base weights passed explicitly
python train.py DINOV3 --unfreeze-layers 4 --checkpoint path/to/dinov3_vitb16_weights.pth
```

Useful variants:

```bash
# Skip the sweep and train directly with the CLI hyperparameters
python train.py DINOV2 --unfreeze-layers 4 --skip-sweep --lr 1e-5 --tau 0.05

# Join an existing sweep (e.g. to add runs from another machine)
python train.py DINOV2 --unfreeze-layers 4 --sweep-id <id>
```

`--unfreeze-layers` is fixed for the whole pipeline (sweep + final training): to compare different unfreezing depths, relaunch with a different value. The per-forward batch size is capped per model to fit in ~12 GB of VRAM and the effective batch is reached via gradient accumulation.

The best checkpoint (by validation PCK@0.10) is saved to `checkpoints/finetune/<model>_unfreeze<N>_best.pth` and can be passed to `eval.py` (DINOv2 and DINOv3; not SAM, see the note above).

## Project structure

```
├── train.py             # two-stage training: W&B sweep (small) + fine-tuning (large)
├── eval.py              # PCK evaluation on the SPair-71k test split
├── sweep_config.yaml    # W&B sweep search space (lr, tau, effective batch)
├── data/
│   └── SPairDataset.py  # SPair-71k pair dataset (images, keypoints, annotations)
├── models/
│   ├── backbone.py      # abstract backbone wrapper (device, patch size, train mode)
│   ├── dino_backbone.py # DINOv2 / DINOv3 wrappers (patch-token feature maps)
│   └── sam.py           # SAM image-encoder wrapper
├── utils/
│   ├── model_builder.py # model + preprocessing factory, torch.compile helper
│   ├── preprocess.py    # resize / normalize / pad pipeline per backbone
│   ├── featuremap.py    # keypoint <-> feature-grid mapping, dense correspondence
│   ├── loss.py          # symmetric dense InfoNCE loss
│   ├── trainer.py       # training loop (InfoNCE, grad accumulation, AMP)
│   ├── evaluator.py     # validation loop (PCK)
│   └── results.py       # PCK aggregation and reporting
└── dinov3-git/          # official DINOv3 repo, cloned by the user (see Setup)
```

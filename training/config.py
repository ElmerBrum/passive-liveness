"""
Training configuration for the MiniFASNet / MultiFTNet fine-tune.

This file is the SHARED CONTRACT between the data-prep, dataset, model,
training and evaluation modules. Field names here are referenced across
all of them — keep them stable.

See docs/08_plano-de-treinamento.md and docs/10_plano-dataset-publico.md.
"""

from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data contract (produced by prepare_public_dataset.py, consumed by dataset.py)
# ---------------------------------------------------------------------------
# Prepared patches live at:
#     <data_root>/<patch_info>/<split>/<class>/*.png
#   patch_info : e.g. "2.7_80x80"  (scale + HxW)
#   split      : "train" | "val" | "test"
#   class      : "0" = spoof/fake , "1" = live/real
#   images     : 80x80, 3-channel, saved with cv2.imwrite (BGR), pixels [0,255]
#
# A manifest CSV accompanies it at <data_root>/manifest.csv with columns:
#     filepath, subject_id, label, pai_type, split
#   label    : 0 = spoof , 1 = live
#   pai_type : "live" or an attack type (e.g. "print", "replay") — metadata,
#              used only for per-PAI APCER reporting in evaluate.py.
#
# LABEL SIGN (the classic bug): CelebA-Spoof uses 0 = live in its own
# annotations, but here folder/label 1 = live/real to match MiniFASNet
# inference (label==1 => real). prepare_public_dataset.py must invert.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TrainConfig:
    # --- data ---
    data_root: str = str(PROJECT_ROOT / "training" / "data")
    patch_info: str = "2.7_80x80"          # which scale/size folder to train on
    input_size: tuple[int, int] = (80, 80)  # (H, W)

    # --- model ---
    # One of: MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE.
    # The original MultiFTNet used MiniFASNetV2SE; we keep that as default.
    model_type: str = "MiniFASNetV2SE"
    num_classes: int = 2                    # 0=spoof, 1=live (binary)
    embedding_size: int = 128
    img_channel: int = 3

    # --- optimisation ---
    lr: float = 1e-3                        # << original 1e-1; lower for fine-tune
    milestones: tuple[int, ...] = (10, 15, 20)
    gamma: float = 0.1
    epochs: int = 25
    momentum: float = 0.9
    weight_decay: float = 5e-4
    batch_size: int = 256                   # RTX 5070 (12GB) handles this at 80x80
    num_workers: int = 8
    ft_loss_weight: float = 0.5             # loss = (1-w)*CE + w*MSE_ft

    # --- runtime ---
    device: str = "cuda"                    # RTX 5070; falls back handled in train.py
    amp: bool = True                        # mixed precision (big speedup on Blackwell)
    seed: int = 42

    # --- outputs ---
    out_dir: str = str(PROJECT_ROOT / "training" / "runs")
    log_every: int = 20                     # steps
    save_every_epochs: int = 5

    # --- derived (filled by __post_init__) ---
    kernel_size: tuple[int, int] = field(default=(0, 0))
    ft_size: tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        h, w = self.input_size
        self.kernel_size = ((h + 15) // 16, (w + 15) // 16)   # get_kernel()
        # Fourier target size = spatial size of the feature map after conv_4,
        # which for 80x80 input is 10x10 (= 2 * kernel_size). See MultiFTNet.
        self.ft_size = (2 * self.kernel_size[0], 2 * self.kernel_size[1])


def get_default_config() -> TrainConfig:
    return TrainConfig()

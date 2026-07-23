# training/

Training pipeline to fine-tune / train MiniFASNet (via MultiFTNet) on a
face anti-spoofing dataset. Phase 0 uses a **public dataset (CelebA-Spoof)**
to validate the pipeline before collecting our own data
(see `../docs/10_plano-dataset-publico.md`).

> ⚠️ AI-assisted code — verify and test before trusting. See repo README.

## Files

| File | Role |
|---|---|
| `config.py` | `TrainConfig` — the **shared contract** (paths, sizes, hyperparams) |
| `models/multiftnet.py` | `MultiFTNet` = MiniFASNet backbone + Fourier aux branch |
| `dataset.py` | `SpoofFTDataset` — loads 80×80 BGR patches + online Fourier target |
| `prepare_public_dataset.py` | CelebA-Spoof → prepared patch folders + manifest |
| `train.py` | training loop (CE + Fourier MSE), AMP, checkpointing |
| `evaluate.py` | APCER / BPCER / ACER, EER threshold on val |

## Data contract (produced by prepare, consumed by dataset/train/eval)

```
training/data/<patch_info>/<split>/<class>/*.png
  patch_info : e.g. 2.7_80x80        (scale_HxW)
  split      : train | val | test
  class      : 0 = spoof/fake , 1 = live/real
  image      : 80x80, 3ch, cv2.imwrite (BGR), pixels [0,255]

training/data/manifest.csv
  columns: filepath, subject_id, label, pai_type, split
  label   : 0 = spoof , 1 = live
  pai_type: "live" | attack type   (metadata, for per-PAI APCER)
```

**Label sign (the classic bug):** CelebA-Spoof annotations use `0 = live`, but
here **folder/label `1` = live/real** to match MiniFASNet inference (`label==1
⇒ real`). `prepare_public_dataset.py` must invert. After training, sanity-check
the sign on known real images (`../images/custom/*` or `../images/sample/*_T*`).

## Model I/O contract

```
MultiFTNet(model_type, num_classes=2, embedding_size=128, conv6_kernel=(5,5))
  input : FloatTensor [B,3,80,80], BGR, [0,255]   (NOT /255)
  train : forward(x) -> (cls_logits[B,2], ft_map[B,1,10,10])
  eval  : forward(x) -> cls_logits[B,2]
  .backbone_state_dict() -> MiniFASNet weights only (for inference/ONNX)
```

## Checkpoint contract (train.py saves, evaluate.py loads)

```python
torch.save({
    "model_state": multiftnet.state_dict(),   # full MultiFTNet
    "backbone_state": multiftnet.backbone_state_dict(),  # MiniFASNet only
    "config": asdict(cfg),
}, path)
```
`live` probability at inference = `softmax(cls_logits)[:, 1]`.

## Quick start

```bash
# 1. prepare (point at the downloaded CelebA-Spoof)
python training/prepare_public_dataset.py --celeba-root /path/to/CelebA_Spoof \
    --out training/data --patch-info 2.7_80x80 --bbox-scale 2.7

# 2. train
python training/train.py           # uses config.py defaults

# 3. evaluate
python training/evaluate.py --checkpoint training/runs/best.pth
```

## Hardware — RTX 5070 (Blackwell, sm_120)

The RTX 50-series needs a **CUDA 12.8+ PyTorch build**; older wheels don't have
`sm_120` kernels and will fail with "no kernel image is available".

```bash
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.is_available())"
```

`config.amp = True` enables mixed precision (big speedup on Blackwell).
Batch size 256 at 80×80 fits comfortably in 12 GB.

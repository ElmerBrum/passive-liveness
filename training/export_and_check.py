#!/usr/bin/env python3
"""
Export a trained checkpoint into the inference pipeline and sanity-check the
LABEL SIGN on known images.

What it does:
  1. Loads a training checkpoint (contract: model_state / backbone_state / config).
  2. Writes the MiniFASNet backbone as a properly-named .pth into an output dir
     (default training/exported_models), so predict.py / webcam.py can use it via
     --models-dir. The name follows the parse_model_name convention:
         <scale>_<HxW>_<ModelType>.pth   e.g. 2.7_80x80_MiniFASNetV2SE.pth
  3. Optionally exports the same backbone to .onnx (for the onnx backend).
  4. SIGN SANITY-CHECK: runs the full inference path (face detect -> scale crop ->
     BGR/[0,255] -> model) on a few known images and prints the LIVE score
     (softmax[:,1]). A correctly-signed model gives HIGH live score on real faces
     and LOW on spoofs. See docs/10_plano-dataset-publico.md (bug #1).

Usage:
    python training/export_and_check.py --checkpoint training/runs/<run>/best.pth
    python training/export_and_check.py --checkpoint ... --scale 2.7 --onnx
    python training/export_and_check.py --checkpoint ... --no-check   # export only

Then test in the real pipeline:
    python predict.py --image images/sample/image_T1.jpg \
        --backend pytorch --models-dir training/exported_models
    python webcam.py --backend pytorch --models-dir training/exported_models
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Make repo root importable (for the liveness package) and this dir (config etc.)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from liveness.model import MODEL_REGISTRY          # noqa: E402
from liveness.utils import get_kernel              # noqa: E402
from liveness.predictor import FaceDetector        # noqa: E402
from liveness.cropper import CropImage             # noqa: E402

DETECTION_DIR = ROOT / "resources" / "detection"
SAMPLE_DIR = ROOT / "images" / "sample"
DEFAULT_OUT = HERE / "exported_models"

# Known samples from the original repo: T = real, F = fake.
DEFAULT_CHECK = [
    ("image_T1.jpg", "real"),
    ("image_F1.jpg", "fake"),
    ("image_F2.jpg", "fake"),
]


def load_backbone(checkpoint_path: Path):
    """Return (backbone_state_dict, model_type, num_classes, patch_hw)."""
    ck = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "backbone_state" not in ck or "config" not in ck:
        sys.exit("ERRO: checkpoint não está no formato esperado "
                 "(faltam 'backbone_state'/'config'). É um checkpoint do train.py?")
    cfg = ck["config"]
    state = ck["backbone_state"]
    model_type = cfg["model_type"]
    num_classes = int(state["prob.weight"].shape[0])
    h, w = cfg.get("input_size", [80, 80])
    return state, model_type, num_classes, (int(h), int(w))


def build_inference_model(model_type: str, num_classes: int, patch_hw):
    kernel = get_kernel(*patch_hw)
    model = MODEL_REGISTRY[model_type](conv6_kernel=kernel, num_classes=num_classes)
    model.eval()
    return model


def export_pth(state, model_type, scale, patch_hw, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = patch_hw
    # Name must satisfy parse_model_name: <scale>_<HxW>_<ModelType>.pth
    name = f"{scale}_{h}x{w}_{model_type}.pth"
    out_path = out_dir / name
    torch.save(state, str(out_path))
    print(f"Backbone salvo → {out_path}")
    return out_path


def export_onnx(model, patch_hw, out_path: Path) -> None:
    h, w = patch_hw
    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, str(out_path),
        opset_version=18, input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    print(f"ONNX salvo → {out_path}")


def live_score(model, detector, cropper, img_bgr, scale, patch_hw) -> tuple[float, int, list]:
    """Full inference path → (live_prob, num_classes, bbox)."""
    bbox = detector.get_bbox(img_bgr)
    h, w = patch_hw
    patch = cropper.crop(org_img=img_bgr, bbox=bbox, scale=scale,
                         out_w=w, out_h=h, crop=True)
    tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0)  # BGR [0,255]
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).numpy()[0]
    return float(probs[1]), probs.shape[0], bbox


def run_sign_check(model, scale, patch_hw, checks) -> bool:
    if not DETECTION_DIR.exists():
        print("AVISO: pasta do detector não encontrada — pulando sign-check.")
        return True
    detector = FaceDetector(DETECTION_DIR)
    cropper = CropImage()

    print("\n" + "=" * 56)
    print("SIGN SANITY-CHECK  (live score = softmax[:,1])")
    print("Esperado: real → alto, fake → baixo")
    print("=" * 56)
    ok = True
    for fname, expected in checks:
        path = SAMPLE_DIR / fname
        img = cv2.imread(str(path))
        if img is None:
            print(f"  {fname:16s}  (não encontrada — pulando)")
            continue
        score, ncls, _ = live_score(model, detector, cropper, img, scale, patch_hw)
        verdict = "real" if score >= 0.5 else "fake"
        flag = "OK" if verdict == expected else "XX  <-- INVERTIDO?"
        print(f"  {fname:16s}  live={score:.4f}  → {verdict:4s} "
              f"(esperado {expected})  {flag}")
        if verdict != expected:
            ok = False
    print("=" * 56)
    if ok:
        print("Sinal OK: real→real, fake→fake. Modelo pronto p/ predict.py/webcam.py.")
    else:
        print("ATENÇÃO: sinal possivelmente invertido ou modelo fraco neste domínio.\n"
              "Lembre: este modelo foi treinado em CelebA-Spoof, não na sua webcam —\n"
              "erros nas amostras do repo podem ser domain shift, não bug de sinal.")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Export + sign sanity-check")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--scale", default="2.7",
                   help="Scale prefix for the exported name / crop (default 2.7)")
    p.add_argument("--onnx", action="store_true", help="Também exportar .onnx")
    p.add_argument("--no-check", action="store_true", help="Só exportar, sem sign-check")
    args = p.parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"ERRO: checkpoint não encontrado: {args.checkpoint}")

    state, model_type, num_classes, patch_hw = load_backbone(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  model_type={model_type}  num_classes={num_classes}  patch={patch_hw}")

    pth_path = export_pth(state, model_type, args.scale, patch_hw, args.out_dir)

    model = build_inference_model(model_type, num_classes, patch_hw)
    model.load_state_dict(state)

    if args.onnx:
        export_onnx(model, patch_hw, pth_path.with_suffix(".onnx"))

    if not args.no_check:
        run_sign_check(model, float(args.scale), patch_hw, DEFAULT_CHECK)


if __name__ == "__main__":
    main()

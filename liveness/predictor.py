"""
Face liveness predictor using MiniFASNet models.

Key differences from the original repo:
 1. torch.load(..., weights_only=False) — explicit flag required in PyTorch >= 2.0
    to silence the FutureWarning about the default changing.
 2. F.softmax(..., dim=1) — dim argument is mandatory in modern PyTorch.
 3. ToTensor keeps pixel values in [0, 255] (NOT divided by 255).
    This matches the training behaviour in the original repo where functional.py
    intentionally removed the .div(255) call. Using torchvision.ToTensor() here
    would silently break inference because it normalises to [0, 1].
 4. Caffe detection model path is now passed explicitly (no hardcoded CWD).
"""

import os
import math
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import MODEL_REGISTRY
from .utils import get_kernel, parse_model_name


class FaceDetector:
    """Thin wrapper around the Caffe RetinaFace model bundled in resources/."""

    CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, detection_model_dir: str | Path):
        detection_model_dir = Path(detection_model_dir)
        prototxt = str(detection_model_dir / "deploy.prototxt")
        caffemodel = str(detection_model_dir / "Widerface-RetinaFace.caffemodel")
        self.net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

    def get_bbox(self, img: np.ndarray) -> list[int]:
        h, w = img.shape[:2]
        aspect = w / h
        if w * h >= 192 * 192:
            img = cv2.resize(img,
                             (int(192 * math.sqrt(aspect)), int(192 / math.sqrt(aspect))),
                             interpolation=cv2.INTER_LINEAR)

        blob = cv2.dnn.blobFromImage(img, 1, mean=(104, 117, 123))
        self.net.setInput(blob, 'data')
        out = self.net.forward('detection_out').squeeze()
        best = np.argmax(out[:, 2])
        left   = out[best, 3] * w
        top    = out[best, 4] * h
        right  = out[best, 5] * w
        bottom = out[best, 6] * h
        return [int(left), int(top), int(right - left + 1), int(bottom - top + 1)]


class LivenessPredictor:
    """
    Runs one or more MiniFASNet models on a cropped face patch and returns a
    fused score.

    Args:
        models_dir:          folder containing .pth weight files.
        detection_model_dir: folder containing deploy.prototxt and .caffemodel.
        device:              torch device string ('cpu', 'cuda:0', …).
    """

    def __init__(self, models_dir: str | Path,
                 detection_model_dir: str | Path,
                 device: str = "cpu"):
        self.device = torch.device(device)
        self.models_dir = Path(models_dir)
        self.detector = FaceDetector(detection_model_dir)
        self._model_cache: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self, model_path: Path) -> torch.nn.Module:
        key = str(model_path)
        if key in self._model_cache:
            return self._model_cache[key]

        h, w, model_type, _ = parse_model_name(model_path.name)
        kernel = get_kernel(h, w)

        # weights_only=False: these checkpoints contain Python objects (OrderedDict).
        # We trust this specific file; in production you'd verify the source.
        state_dict = torch.load(str(model_path), map_location=self.device,
                                weights_only=False)

        # Strip DataParallel "module." prefix if present
        first_key = next(iter(state_dict))
        if first_key.startswith('module.'):
            state_dict = OrderedDict(
                (k[len('module.'):], v) for k, v in state_dict.items()
            )

        # Infer num_classes from the classifier head shape so both the original
        # 3-class models and our fine-tuned binary (2-class) models load. The
        # final layer is `prob = Linear(embedding, num_classes)` → 'prob.weight'
        # has shape (num_classes, embedding).
        num_classes = state_dict["prob.weight"].shape[0]
        model = MODEL_REGISTRY[model_type](
            conv6_kernel=kernel, num_classes=num_classes).to(self.device)

        model.load_state_dict(state_dict)
        model.eval()
        self._model_cache[key] = model
        return model

    @staticmethod
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        """
        Convert an OpenCV BGR image (H×W×C uint8) to a float32 tensor (C×H×W).
        Values are kept in [0, 255] — NOT normalised to [0, 1] — because the
        original MiniFASNet weights were trained this way.
        """
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
        return tensor.unsqueeze(0)  # add batch dim → (1, C, H, W)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_bbox(self, img: np.ndarray) -> list[int]:
        return self.detector.get_bbox(img)

    def predict(self, img: np.ndarray, model_path: Path) -> np.ndarray:
        """Return softmax probabilities for a single model."""
        tensor = self._to_tensor(img).to(self.device)
        model = self._load_model(model_path)
        with torch.no_grad():
            logits = model(tensor)
            # dim=1 is required in modern PyTorch; the original omitted it,
            # which triggered a deprecation warning and may misbehave.
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict_ensemble(self, img: np.ndarray,
                         cropper,
                         threshold: float = 0.7) -> tuple[int, float, list[int], list[dict]]:
        """
        Run all models in models_dir and fuse by summing their softmax outputs.

        Returns:
            label (int):   1=real, 0=fake, -1=inconclusive (score < threshold)
            score (float): média da probabilidade da classe vencedora em [0, 1]
            bbox (list):   [x, y, w, h] of the detected face
            per_model (list[dict]): scores individuais de cada modelo
        """
        bbox = self.get_bbox(img)
        pth_files = sorted(self.models_dir.glob("*.pth"))
        fused = None                       # sized to the first model's class count
        per_model = []

        for pth in pth_files:
            h, w, _, scale = parse_model_name(pth.name)
            patch = cropper.crop(
                org_img=img, bbox=bbox,
                scale=scale if scale is not None else 1.0,
                out_w=w, out_h=h,
                crop=(scale is not None),
            )
            probs = self.predict(patch, pth)
            fused = probs if fused is None else fused + probs
            per_model.append({
                "name": pth.stem,
                "fake": float(probs[0, 0]),
                "real": float(probs[0, 1]),
            })

        # Só considera classes 0 (fake) e 1 (real) — classe 2 é artefato de treino.
        # Ver docs/07_uso-dos-modelos.md
        winner = int(np.argmax(fused[:, :2]))
        score  = float(fused[0, winner] / len(pth_files))
        label  = winner if score >= threshold else -1
        return label, score, bbox, per_model

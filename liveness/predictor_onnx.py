"""
Predictor de liveness usando onnxruntime — sem dependência do PyTorch em runtime.

Diferenças em relação ao predictor PyTorch (predictor.py):
  - Carrega os modelos .onnx (não .pth)
  - Usa onnxruntime.InferenceSession para inferência
  - Softmax aplicado com numpy (scipy ou manual) — sem torch.nn.functional
  - Detector facial (Caffe/OpenCV) é idêntico — reusa FaceDetector de predictor.py

Por que isso é útil:
  - Deploy sem instalar torch (muito menor em disco/RAM)
  - Compatível com runtimes de edge (ONNX Runtime Mobile, TensorRT, etc.)
  - Inferência pode ser mais rápida com providers (CUDA, DirectML, CoreML)
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from .predictor import FaceDetector   # reusa detector; evita duplicação
from .utils import parse_model_name


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class LivenessPredictorONNX:
    """
    Predictor de liveness usando exclusivamente onnxruntime.

    Args:
        models_dir:          pasta com os arquivos .onnx
        detection_model_dir: pasta com deploy.prototxt e .caffemodel
        providers:           lista de execution providers do onnxruntime.
                             Default: ['CPUExecutionProvider'].
                             Para GPU: ['CUDAExecutionProvider', 'CPUExecutionProvider']
    """

    def __init__(self, models_dir: str | Path,
                 detection_model_dir: str | Path,
                 providers: list[str] | None = None):
        self.models_dir = Path(models_dir)
        self.detector = FaceDetector(detection_model_dir)
        self.providers = providers or ["CPUExecutionProvider"]
        self._sessions: dict[str, ort.InferenceSession] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _session(self, onnx_path: Path) -> ort.InferenceSession:
        key = str(onnx_path)
        if key not in self._sessions:
            self._sessions[key] = ort.InferenceSession(
                str(onnx_path), providers=self.providers
            )
        return self._sessions[key]

    @staticmethod
    def _to_numpy(img: np.ndarray) -> np.ndarray:
        """
        BGR uint8 (H, W, 3) → float32 (1, 3, H, W) em [0, 255].
        Mesma escala do predictor PyTorch — ver docs/01_pixel-range-0-255.md.
        """
        tensor = img.transpose(2, 0, 1).astype(np.float32)
        return np.expand_dims(tensor, axis=0)   # adiciona dimensão de batch

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def get_bbox(self, img: np.ndarray) -> list[int]:
        return self.detector.get_bbox(img)

    def predict(self, img: np.ndarray, onnx_path: Path) -> np.ndarray:
        """Retorna probabilidades softmax (1, num_classes) para um modelo."""
        session = self._session(onnx_path)
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: self._to_numpy(img)})[0]
        return _softmax(logits)

    def predict_ensemble(self, img: np.ndarray,
                         cropper,
                         threshold: float = 0.7) -> tuple[int, float, list[int], list[dict]]:
        """
        Roda todos os modelos .onnx em ensemble e retorna (label, score, bbox, per_model).
        label: 1=real, 0=fake, -1=inconclusivo (score < threshold).
        per_model: lista de dicts com name, fake, real para cada modelo.
        """
        bbox = self.get_bbox(img)
        onnx_files = sorted(self.models_dir.glob("*.onnx"))
        fused = np.zeros((1, 3), dtype=np.float32)
        per_model = []

        for onnx_path in onnx_files:
            h, w, _, scale = parse_model_name(onnx_path.name)
            patch = cropper.crop(
                org_img=img, bbox=bbox,
                scale=scale if scale is not None else 1.0,
                out_w=w, out_h=h,
                crop=(scale is not None),
            )
            probs = self.predict(patch, onnx_path)
            fused += probs
            per_model.append({
                "name": onnx_path.stem,
                "fake": float(probs[0, 0]),
                "real": float(probs[0, 1]),
            })

        # Só considera classes 0 (fake) e 1 (real) — classe 2 é artefato de treino.
        # Ver docs/07_uso-dos-modelos.md
        winner = int(np.argmax(fused[:, :2]))
        score  = float(fused[0, winner] / len(onnx_files))
        label  = winner if score >= threshold else -1
        return label, score, bbox, per_model

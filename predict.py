#!/usr/bin/env python3
"""
Detecção passiva de liveness com MiniFASNet.

Exemplos de uso:
    # Imagem de amostra do repo (backend padrão: pytorch)
    python predict.py --image images/sample/image_T1.jpg

    # Backend ONNX (não requer torch em runtime)
    python predict.py --image images/sample/image_T1.jpg --backend onnx

    # Salvar imagem anotada com bbox e score
    python predict.py --image images/sample/image_F1.jpg --save

    # Imagem própria
    python predict.py --image images/custom/minha_foto.jpg --save

    # GPU (apenas backend pytorch)
    python predict.py --image images/sample/image_T1.jpg --device cuda:0

Requisito de proporção:
    O modelo foi treinado em frames de câmera Android (proporção 3:4, w:h).
    Imagens com proporção diferente funcionam mas o score pode ser menos confiável.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT  = Path(__file__).parent
MODELS_DIR    = PROJECT_ROOT / "resources" / "models"
DETECTION_DIR = PROJECT_ROOT / "resources" / "detection"


def check_aspect_ratio(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    ratio = w / h
    if abs(ratio - 3 / 4) > 0.05:
        warnings.warn(
            f"Proporção da imagem: {ratio:.2f} (w/h), esperado ~0.75 (3:4). "
            "Resultados podem ser menos precisos."
        )


def annotate_and_save(img: np.ndarray, bbox: list, label: int, score: float,
                      output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    color      = (255, 0, 0) if label == 1 else (0, 0, 255)
    tag        = f"{'Real' if label == 1 else 'Fake'} Face  {score:.2f}"
    font_scale = 0.5 * img.shape[0] / 1024

    cv2.rectangle(img,
                  (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  color, 2)
    cv2.putText(img, tag, (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, font_scale, color)
    cv2.imwrite(str(output_path), img)
    print(f"Resultado salvo → {output_path}")


def run(image_path: Path, backend: str, device: str, save: bool,
        output_dir: Path | None = None, threshold: float = 0.7,
        models_dir: Path = MODELS_DIR) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        sys.exit(f"ERRO: não foi possível ler a imagem: {image_path}")

    check_aspect_ratio(img)

    from liveness.cropper import CropImage
    cropper = CropImage()

    if backend == "onnx":
        from liveness.predictor_onnx import LivenessPredictorONNX
        predictor = LivenessPredictorONNX(
            models_dir=models_dir,
            detection_model_dir=DETECTION_DIR,
        )
    else:
        from liveness.predictor import LivenessPredictor
        predictor = LivenessPredictor(
            models_dir=models_dir,
            detection_model_dir=DETECTION_DIR,
            device=device,
        )

    t0 = time.perf_counter()
    result = predictor.predict_ensemble(img, cropper, threshold=threshold)
    elapsed = time.perf_counter() - t0

    label, score, bbox = result[0], result[1], result[2]
    per_model = result[3] if len(result) > 3 else None

    verdict = {1: "Real Face", 0: "Fake Face"}.get(label, "Inconclusive")
    print(f"\nImagem  : {image_path.name}")
    print(f"Backend : {backend}")
    print(f"Result  : {verdict}")
    print(f"Score   : {score:.4f}")
    if per_model:
        print(f"Modelos :")
        for m in per_model:
            print(f"  {m['name']}")
            print(f"    fake={m['fake']:.4f}  real={m['real']:.4f}")
    print(f"BBox    : x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    print(f"Tempo   : {elapsed:.3f}s\n")

    if save:
        dest = output_dir if output_dir else image_path.parent
        out  = dest / f"{image_path.stem}_result{image_path.suffix}"
        annotate_and_save(img.copy(), bbox, label, score, out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Liveness passivo — MiniFASNet")
    parser.add_argument("--image",   required=True, type=Path,
                        help="Caminho para a imagem de entrada")
    parser.add_argument("--backend", default="pytorch", choices=["pytorch", "onnx"],
                        help="pytorch (default) ou onnx")
    parser.add_argument("--save",    action="store_true",
                        help="Salvar imagem anotada com bbox e score")
    parser.add_argument("--device",  default="cpu",
                        help="Dispositivo torch: cpu | cuda:0  (só backend pytorch)")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Pasta de destino para imagens anotadas (default: mesma da imagem)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Score mínimo para aceitar a decisão (default: 0.7). "
                             "Abaixo disso o resultado é 'Inconclusive'.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR,
                        help="Pasta com os modelos (.pth/.onnx). "
                             "Default: resources/models. Use training/exported_models "
                             "para testar um modelo treinado.")
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f"ERRO: imagem não encontrada: {args.image}")

    run(args.image, args.backend, args.device, args.save, args.output_dir,
        args.threshold, args.models_dir)


if __name__ == "__main__":
    main()

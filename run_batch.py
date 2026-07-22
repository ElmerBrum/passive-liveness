#!/usr/bin/env python3
"""
Roda predict.py em todas as imagens de uma pasta e consolida os outputs.

Uso:
    python run_batch.py
    python run_batch.py --input-dir images/custom --threshold 0.6
    python run_batch.py --input-dir images/sample --backend pytorch
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
EXTENSIONS   = {".jpg", ".jpeg", ".png", ".webp"}


def run_one(image: Path, backend: str, threshold: float,
            output_dir: Path) -> str:
    cmd = [
        sys.executable, str(PROJECT_ROOT / "predict.py"),
        "--image",      str(image),
        "--backend",    backend,
        "--threshold",  str(threshold),
        "--save",
        "--output-dir", str(output_dir),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch liveness — roda todas as imagens de uma pasta")
    parser.add_argument("--input-dir",  type=Path, default=PROJECT_ROOT / "images" / "custom",
                        help="Pasta com as imagens de entrada")
    parser.add_argument("--backend",    default="onnx", choices=["pytorch", "onnx"])
    parser.add_argument("--threshold",  type=float, default=0.7)
    args = parser.parse_args()

    input_dir  = args.input_dir.resolve()
    output_dir = input_dir / "results"
    output_dir.mkdir(exist_ok=True)

    images = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTENSIONS
    )

    if not images:
        sys.exit(f"Nenhuma imagem encontrada em {input_dir}")

    print(f"Encontradas {len(images)} imagem(ns) em {input_dir}")
    print(f"Backend: {args.backend}  |  Limiar: {args.threshold}\n")

    outputs = []
    for img in images:
        print(f"  → {img.name}")
        out = run_one(img, args.backend, args.threshold, output_dir)
        outputs.append((img.name, out))

    # Concatena tudo em um arquivo único
    summary_path = output_dir / "all_results.txt"
    sep = "-" * 60
    with summary_path.open("w") as f:
        f.write(f"Batch liveness — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Backend: {args.backend}  |  Limiar: {args.threshold}\n")
        f.write(f"Total: {len(images)} imagem(ns)\n")
        f.write(f"{sep}\n\n")
        for name, out in outputs:
            f.write(f"[ {name} ]\n{out}\n\n{sep}\n\n")

    print(f"\nImagens anotadas → {output_dir}/")
    print(f"Consolidado      → {summary_path}")


if __name__ == "__main__":
    main()

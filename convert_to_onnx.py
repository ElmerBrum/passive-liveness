#!/usr/bin/env python3
"""
Converte os modelos MiniFASNet de .pth para .onnx.

Os arquivos .onnx são salvos na mesma pasta dos .pth, mantendo o mesmo
prefixo de nome para que parse_model_name() continue funcionando.

Uso:
    python convert_to_onnx.py

Saída esperada:
    resources/models/2.7_80x80_MiniFASNetV2.onnx
    resources/models/4_0_0_80x80_MiniFASNetV1SE.onnx

Notas sobre opset:
    torch.onnx.export com dynamo=False (exporter legado) suporta até opset 20.
    Usamos 18 porque é o que o novo exporter do torch 2.x gera nativamente e
    porque onnxruntime >= 1.16 suporta opset 18. Ver docs/06_onnx-conversion.md.
"""

import warnings
from pathlib import Path
from collections import OrderedDict

import torch
import onnx

from liveness.model import MODEL_REGISTRY
from liveness.utils import parse_model_name, get_kernel

MODELS_DIR = Path(__file__).parent / "resources" / "models"
OPSET = 18


def load_pytorch_model(pth_path: Path, device: torch.device) -> torch.nn.Module:
    h, w, model_type, _ = parse_model_name(pth_path.name)
    kernel = get_kernel(h, w)
    model = MODEL_REGISTRY[model_type](conv6_kernel=kernel).to(device)

    state_dict = torch.load(str(pth_path), map_location=device, weights_only=False)
    first_key = next(iter(state_dict))
    if first_key.startswith('module.'):
        state_dict = OrderedDict((k[len('module.'):], v) for k, v in state_dict.items())

    model.load_state_dict(state_dict)
    model.eval()
    return model


def export(pth_path: Path, device: torch.device) -> Path:
    h, w, _, _ = parse_model_name(pth_path.name)
    onnx_path = pth_path.with_suffix('.onnx')

    print(f"  Carregando  {pth_path.name}")
    model = load_pytorch_model(pth_path, device)

    # Input: (batch, 3, h, w) float32 em [0, 255] — ver docs/01_pixel-range-0-255.md
    dummy = torch.zeros(1, 3, h, w, dtype=torch.float32, device=device)

    print(f"  Exportando  {onnx_path.name}  (opset {OPSET})")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=OPSET,
            input_names=["input"],
            output_names=["logits"],   # logits brutos; softmax aplicado na inferência
            dynamic_axes={
                "input":  {0: "batch"},
                "logits": {0: "batch"},
            },
            dynamo=False,              # exporter legado: mais estável, sem onnxscript
        )

    # Valida estrutura do grafo (shapes, ops)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    print(f"  Validado OK → {onnx_path.relative_to(Path.cwd())}")
    return onnx_path


def main() -> None:
    device = torch.device("cpu")   # CPU garante portabilidade máxima do grafo
    pth_files = sorted(MODELS_DIR.glob("*.pth"))

    if not pth_files:
        print(f"Nenhum .pth encontrado em {MODELS_DIR}")
        return

    # Remove .onnx existentes para garantir conversão limpa
    for f in MODELS_DIR.glob("*.onnx"):
        f.unlink()

    print(f"Convertendo {len(pth_files)} modelo(s)...\n")
    for pth in pth_files:
        export(pth, device)
        print()

    print("Concluído.")


if __name__ == "__main__":
    main()

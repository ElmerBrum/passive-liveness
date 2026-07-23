# Docs — índice

> ⚠️ **Conteúdo gerado com apoio de IA** — pode conter erros ou estar incompleto,
> especialmente sobre datasets, licenças e detalhes de treinamento. Questione,
> confira as fontes citadas e teste antes de confiar.

Cada arquivo documenta uma decisão ou mudança específica feita ao adaptar
o repo original para Python/PyTorch modernos.

| Arquivo | Tema |
|---|---|
| `00_project-overview.md` | Estrutura geral, o que foi copiado e o que foi descartado |
| `01_pixel-range-0-255.md` | Por que os pixels ficam em [0,255] e não [0,1] |
| `02_pytorch-compat.md` | `F.softmax(dim=)` e `torch.load(weights_only=)` |
| `03_opencv-caffe-removal.md` | OpenCV 5 removeu Caffe; pinamos 4.x |
| `04_model-name-parsing.md` | Como o nome do .pth codifica metadados; bug corrigido |
| `05_ensemble-e-scores.md` | Como os dois modelos são combinados e como ler os scores |
| `06_onnx-conversion.md` | Conversão .pth → .onnx: opset, dynamo, equivalência |
| `07_uso-dos-modelos.md` | Referência técnica: input/output, arquiteturas, como chamar |
| `08_plano-de-treinamento.md` | Plano completo: coleta, dataset, fine-tuning, métricas |
| `09_plano-de-captura-dataset.md` | Metodologia de captura: matriz de variação, PAIs, split, anti-viés |
| `10_plano-dataset-publico.md` | Validar o pipeline com CelebA-Spoof antes da coleta própria |

## Convenção

- Um arquivo por **grande mudança ou decisão de design**.
- Incluir sempre: o que era antes, o problema, o que foi feito, e pontos de atenção futura.
- Numerar em ordem cronológica de descoberta.

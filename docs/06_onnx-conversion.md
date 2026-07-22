# Conversão dos modelos para ONNX

## Por que converter para ONNX

| | PyTorch (.pth) | ONNX (.onnx) |
|---|---|---|
| Runtime necessário | `torch` (~2 GB) | `onnxruntime` (~20 MB) |
| Portabilidade | Depende de versão do torch | Qualquer runtime ONNX |
| Velocidade (CPU) | baseline | ~2× mais rápido (medido) |
| Deploy em edge | Complexo | TensorRT, CoreML, DirectML, Mobile |
| Treinamento | Sim | Não |

## O que o script `convert_to_onnx.py` faz

1. Carrega o `.pth` com `torch.load(..., weights_only=False)`
2. Cria um tensor dummy `(1, 3, 80, 80)` float32 em [0, 255]
3. Chama `torch.onnx.export(..., dynamo=False)` — exporter legado, estável
4. Valida o grafo com `onnx.checker.check_model()`
5. Salva o `.onnx` na mesma pasta do `.pth`, mesmo prefixo de nome

```
resources/models/
├── 2.7_80x80_MiniFASNetV2.pth
├── 2.7_80x80_MiniFASNetV2.onnx     ← gerado
├── 4_0_0_80x80_MiniFASNetV1SE.pth
└── 4_0_0_80x80_MiniFASNetV1SE.onnx ← gerado
```

## Detalhes técnicos do grafo ONNX

**Opset 18** — não opset 11 como o código original tentava.

Motivo: `torch.onnx.export` no torch >= 2.x usa opset 18 internamente.
Tentar converter para opset 11 retroativamente falha porque ops como `GroupNormalization`
e `BatchNormalization` tiveram mudanças incompatíveis de versão.
onnxruntime >= 1.16 suporta opset 18 sem problemas.

**Input/output do grafo:**

```
Input:  "input"   → float32  (batch, 3, 80, 80)  valores em [0, 255]
Output: "logits"  → float32  (batch, 3)           logits brutos (pré-softmax)
```

O softmax é aplicado **fora** do grafo ONNX, no `predictor_onnx.py`.
Isso facilita substituir o softmax por temperatura ou outros ajustes sem reconverter.

**Batch dimension dinâmica:**

`dynamic_axes` deixa a dimensão batch variável no grafo.
Na prática sempre usamos batch=1 (uma face por vez), mas permite
experimentar batch > 1 sem reconverter.

## Por que `dynamo=False`

torch >= 2.x tem dois exporters:
- **dynamo=True** (novo, padrão): usa `torch.export` + onnxscript. Mais robusto para
  modelos complexos, mas ainda experimental para muitos casos.
- **dynamo=False** (legado): usa `torch.jit.trace`. Mais estável para redes simples
  como MiniFASNet. Produz grafos mais limpos.

Usamos `dynamo=False` porque MiniFASNet é uma rede feed-forward sem fluxo condicional
dinâmico — o exporter legado é suficiente e mais previsível.

## Como usar o backend ONNX

```bash
# Inferência com onnxruntime (sem torch)
python predict.py --image images/sample/image_T1.jpg --backend onnx

# Para GPU com onnxruntime-gpu
# pip install onnxruntime-gpu
python predict.py --image images/sample/image_T1.jpg --backend onnx
# (o provider CUDA é detectado automaticamente se instalado)
```

## Resultados de equivalência (CPU)

| Imagem | PyTorch score | ONNX score | Tempo PyTorch | Tempo ONNX |
|---|---|---|---|---|
| image_T1.jpg (real) | 0.9936 | 0.9936 | 0.076s | 0.038s |
| image_F1.jpg (fake) | 0.7316 | 0.7316 | 0.075s | 0.034s |
| image_F2.jpg (fake) | 0.8172 | 0.8172 | 0.075s | 0.053s |

Scores idênticos confirmam que a conversão preservou os pesos corretamente.

## Pontos de atenção

- Os `.onnx` têm ~1.8 MB cada — similares aos `.pth`.
- O grafo ONNX é **estático** (pesos embutidos). Não é possível fazer fine-tuning
  a partir do `.onnx`; sempre parta do `.pth` para retreinar.
- Para produção, considere proteger os arquivos `.onnx` com criptografia,
  pois os pesos ficam legíveis por qualquer ferramenta ONNX.

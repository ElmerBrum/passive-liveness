# Uso dos modelos — referência técnica

## Modelos disponíveis

| Arquivo | Arquitetura | Parâmetros | Patch | Scale | Classes |
|---|---|---|---|---|---|
| `2.7_80x80_MiniFASNetV2.pth/.onnx`      | MiniFASNetV2   | ~1.8M | 80×80 | 2.7× | 3 |
| `4_0_0_80x80_MiniFASNetV1SE.pth/.onnx`  | MiniFASNetV1SE | ~1.8M | 80×80 | 4.0× | 3 |

## Formato de entrada

```
Tensor: float32  shape (1, 3, H, W)
H, W  : 80 × 80 (fixo para esses modelos)
Canais: BGR  (ordem do OpenCV, não RGB)
Range : [0, 255]  — NÃO normalizado para [0, 1]
```

**Por que BGR e não RGB?**
A imagem vem do OpenCV (`cv2.imread`) que lê em BGR por padrão.
O original não fazia conversão, então o modelo foi treinado com BGR.
Converter para RGB antes de passar ao modelo produziria scores errados.

**Por que [0, 255]?**
Ver `docs/01_pixel-range-0-255.md`.

## Formato de saída

```
Tensor: float32  shape (1, 3)
                 ↑ logits brutos (pré-softmax)

Após softmax:
  [0] → probabilidade de fake  (foto, tela, máscara)
  [1] → probabilidade de real  (rosto ao vivo)
  [2] → não usado na inferência (artefato de treino multi-task)
```

A decisão é `argmax(softmax(logits))`:
- índice 1 = Real
- índice 0 = Fake

## Sobre as arquiteturas

### MiniFASNetV2

Baseada em MobileNetV2 com blocos depthwise separáveis.
Backbone: `MiniFASNet` com configuração de canais `keep_dict['1.8M_']`.
Sem Squeeze-and-Excitation (SE).

### MiniFASNetV1SE

Mesma estrutura do V1 mas com blocos **SE (Squeeze-and-Excitation)** no
último bloco de cada estágio residual.

O SE recalibra os canais via atenção global:
```
x → AvgPool → FC(c → c/r) → ReLU → FC(c/r → c) → Sigmoid → x * gate
```
Isso permite ao modelo focar nos canais mais informativos por contexto.
Melhora a capacidade de distinguir texturas de pele vs papel/tela.

## O que é `scale` e por que importa

O `scale` define o quanto da imagem ao redor do rosto é capturado no patch.

```
bbox detectado: x, y, w, h
patch real:     centro do bbox, mas com dimensões w*scale × h*scale
```

- **scale = 2.7**: patch ~2.7× o rosto — inclui testa, queixo e um pouco de fundo
- **scale = 4.0**: patch ~4.0× o rosto — inclui mais contexto (bordas de papel, tela)

O modelo com scale 4.0 é melhor para detectar fotos impressas (onde os bordos são visíveis).
O com scale 2.7 é mais robusto a enquadramentos apertados.

## Ensemble

Os dois modelos são complementares (arquiteturas diferentes + scales diferentes).
A fusão é feita por **soma de probabilidades softmax**:

```python
fused = predict(patch_27, model_v2) + predict(patch_40, model_v1se)
label = argmax(fused)        # 0=fake, 1=real
score = fused[label] / 2     # média normalizada em [0, 1]
```

## Como chamar diretamente (sem o script predict.py)

### Backend PyTorch

```python
from pathlib import Path
from liveness.predictor import LivenessPredictor
from liveness.cropper import CropImage
import cv2

predictor = LivenessPredictor(
    models_dir="resources/models",
    detection_model_dir="resources/detection",
    device="cpu",
)
img = cv2.imread("images/sample/image_T1.jpg")
label, score, bbox = predictor.predict_ensemble(img, CropImage())
print(label, score)  # 1  0.9936
```

### Backend ONNX

```python
from liveness.predictor_onnx import LivenessPredictorONNX
from liveness.cropper import CropImage
import cv2

predictor = LivenessPredictorONNX(
    models_dir="resources/models",
    detection_model_dir="resources/detection",
)
img = cv2.imread("images/sample/image_T1.jpg")
label, score, bbox = predictor.predict_ensemble(img, CropImage())
print(label, score)  # 1  0.9936
```

## Limitações conhecidas

- Treinado em smartphone Android (câmera frontal, iluminação controlada).
  Performance pode cair em ambientes com iluminação ruim ou câmeras de baixa qualidade.
- Não detecta ataques 3D (máscaras de silicone realistas).
- Proporção 3:4 esperada — imagens muito diferentes podem afetar a detecção facial.
- Um único frame: não há análise temporal de movimento (liveness ativo).

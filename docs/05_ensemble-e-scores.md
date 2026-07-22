# Como o ensemble funciona e o que os scores significam

## Dois modelos, uma decisão

O sistema carrega **dois** modelos de `resources/models/`:

| Modelo | Arquitetura | Scale |
|---|---|---|
| `2.7_80x80_MiniFASNetV2.pth`   | MiniFASNetV2   | 2.7 (contexto médio) |
| `4_0_0_80x80_MiniFASNetV1SE.pth` | MiniFASNetV1SE | 4.0 (contexto amplo) |

Para cada modelo:
1. Recorta um patch da imagem original usando o scale correspondente
2. Redimensiona para 80×80
3. Passa pela rede → logits com shape `(1, 3)`
4. Aplica softmax → probabilidades `[p_fake, p_real, p_unused]`

Depois soma as probabilidades dos dois modelos:

```python
fused = np.zeros((1, 3))
for model in models:
    fused += predictor.predict(patch, model)   # soma acumulada

label = np.argmax(fused)         # 0=fake, 1=real
score = fused[0, label] / n_models  # média normalizada
```

## Por que somar (não fazer média direto)?

Na implementação original o score final é `prediction[0][label] / 2`, o que é
equivalente à média. A soma antes do argmax garante que a decisão seja baseada
na confiança acumulada dos dois modelos — se um diz "real com 0.9" e o outro
diz "fake com 0.6", a soma dá `real=0.9, fake=0.6` → decide real.

## Interpretação dos valores

| Score | Interpretação |
|---|---|
| > 0.9 | Alta confiança |
| 0.6 – 0.9 | Confiança moderada |
| ~0.5 | Incerto (próximo do limiar) |
| < 0.5 | Raramente acontece (argmax garante label ≥ 0.5 da soma) |

## Resultados nas amostras do repo

| Imagem | Resultado | Score |
|---|---|---|
| `image_T1.jpg` (rosto real) | Real Face | 0.9936 |
| `image_F1.jpg` (foto impressa) | Fake Face | 0.7316 |
| `image_F2.jpg` (foto impressa) | Fake Face | 0.8172 |

## A proporção 3:4

O modelo foi treinado em frames de câmera Android (proporção 3:4, landscape).
Imagens com proporção muito diferente podem ter scores menos confiáveis.
O `predict.py` emite um `warnings.warn` se a proporção estiver fora de ±5%.

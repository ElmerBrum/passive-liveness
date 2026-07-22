# Pixels em [0, 255] — não normalizados

## O problema

No repo original existia `src/data_io/functional.py` com a função `to_tensor()`:

```python
def to_tensor(pic):
    if isinstance(pic, np.ndarray):
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        # return img.float().div(255)  modify by zkx
        return img.float()          # ← retorna [0, 255], NÃO [0, 1]
```

A linha `div(255)` foi **comentada intencionalmente** (comentário diz "modify by zkx").

## Por que isso importa

Toda rede neural lida com a escala que viu durante o treino.
Se o modelo foi treinado com pixels em [0, 255] e você passar [0, 1] na inferência,
os scores serão incorretos mas nenhum erro será lançado — falha silenciosa.

O equivalente "padrão" `torchvision.transforms.ToTensor()` **divide por 255**.
Usá-lo aqui quebraria a inferência silenciosamente.

## O que fizemos

Descartamos os 500+ linhas de `transform.py` e `functional.py` e centralizamos
a lógica em 3 linhas com comentário explícito em `liveness/predictor.py`:

```python
@staticmethod
def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """
    Valores mantidos em [0, 255] — NÃO normalizados para [0, 1] — porque
    os pesos MiniFASNet foram treinados com essa escala.
    """
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return tensor.unsqueeze(0)
```

## Como verificar

Você pode confirmar isso olhando o score de saída:
- Com pixels em [0, 255]: score ~0.99 para rosto real
- Com pixels em [0, 1]:  score ~0.33 (distribuição uniforme — modelo confuso)

## Ponto de atenção para o futuro

Se você retreinar o modelo com normalização padrão ([0,1] ou z-score),
precisará atualizar `_to_tensor()`. Documente isso ao treinar.

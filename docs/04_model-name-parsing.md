# Parsing do nome dos arquivos de modelo

## Convenção de nomenclatura

Os arquivos `.pth` codificam metadados no próprio nome:

```
<scale>_<h>x<w>_<ModelType>.pth
```

Exemplos reais:
```
2.7_80x80_MiniFASNetV2.pth        → scale=2.7, h=80, w=80, tipo=MiniFASNetV2
4_0_0_80x80_MiniFASNetV1SE.pth   → scale=4.0, h=80, w=80, tipo=MiniFASNetV1SE
```

**Por que `4_0_0`?** O underscore é o separador do padrão de nomenclatura da Minivision.
`4.0.0` seria versão de software, então eles usaram `4_0_0` para representar o scale `4.0`.
Apenas o **primeiro token** (`4`) é o scale — os demais (`0`, `0`) são ignorados.

## O bug na minha primeira versão

Tentei extrair o scale com regex e depois `replace('_', '.')`:

```python
prefix = stem[: match.start()].rstrip('_')    # "4_0_0"
scale = float(prefix.replace('_', '.'))        # "4.0.0" → ValueError!
```

`float("4.0.0")` lança `ValueError` porque não é um número válido.

## Solução correta (replicando o original)

```python
parts = filename.split('_')          # ['4', '0', '0', '80x80', 'MiniFASNetV1SE.pth']
info  = parts[:-1]                   # ['4', '0', '0', '80x80']  (drop o tipo)
model_type = os.path.splitext(parts[-1])[0]   # 'MiniFASNetV1SE'
h_input, w_input = info[-1].split('x')        # '80x80' → (80, 80)
scale = None if info[0] == 'org' else float(info[0])   # '4' → 4.0
```

A lógica do original era simples: `info[0]` é sempre o scale como string única.
Funciona para `2.7` (um token) e `4` (um token) igualmente.

## O que `scale` significa geometricamente

O `scale` é usado em `CropImage.crop()` para expandir a bounding box do rosto.
Com scale=2.7, o patch recortado é 2.7× maior que o bbox detectado — captura mais
contexto ao redor do rosto (útil para detectar impressões em papel).
Com scale=4.0, o contexto é ainda maior.

O modelo foi treinado com cada scale específico, então importar o scale errado
dará resultados incorretos silenciosamente.

## O que `get_kernel` calcula

```python
def get_kernel(height, width):
    return ((height + 15) // 16, (width + 15) // 16)
```

Para 80×80: `(80+15)//16 = 5` → kernel `(5, 5)`.

Isso define o tamanho do kernel da última camada convolucional (`conv6_dw`).
A camada recebe um feature map de tamanho `(h/16, w/16)` depois das convoluções
e o kernel é exatamente desse tamanho — equivale a um "global average" aprendido.

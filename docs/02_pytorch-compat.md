# Compatibilidade com PyTorch moderno (>= 2.0)

Duas mudanças de API que causam warnings ou erros em versões recentes.

---

## 1. `F.softmax()` sem `dim`

### Original
```python
result = F.softmax(result).cpu().numpy()
```

### Problema
Em PyTorch >= 1.x, omitir `dim` gera:
```
UserWarning: Implicit dimension choice for softmax has been deprecated.
```
Em versões futuras pode virar erro ou mudar de comportamento.

### Correção
```python
probs = F.softmax(logits, dim=1).cpu().numpy()
```

`dim=1` porque o tensor tem shape `(1, num_classes)` —
a dimensão das classes é a 1 (batch é a 0).

---

## 2. `torch.load()` sem `weights_only`

### Original
```python
state_dict = torch.load(model_path, map_location=self.device)
```

### Problema
PyTorch >= 2.0 introduziu o parâmetro `weights_only` e avisa que o default
vai mudar de `False` para `True` em versões futuras:
```
FutureWarning: You are using `torch.load` with `weights_only=False`,
which is the current default...
```
Com `weights_only=True` e um checkpoint antigo (que contém `OrderedDict` de Python),
você recebe erro imediato.

### Correção
```python
state_dict = torch.load(str(model_path), map_location=self.device,
                        weights_only=False)
```

Passamos `False` explicitamente porque:
1. Os `.pth` pré-treinados contêm objetos Python serializados (não só tensors).
2. Confiamos nos arquivos que nós mesmos colocamos em `resources/models/`.

### Ponto de atenção
`weights_only=False` executa código Python arbitrário do checkpoint.
Nunca use com arquivos baixados de fontes desconhecidas.
Para produção considere converter para formato seguro (TorchScript ou ONNX).

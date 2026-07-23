# our-liveness

> ⚠️ **Aviso — conteúdo gerado com apoio de IA.**
> Parte desta documentação (pasta [`docs/`](docs/)) e dos planos de
> desenvolvimento foi produzida com assistência de IA. Pode conter informações
> **incompletas, desatualizadas ou incorretas** — inclusive sobre datasets,
> licenças, protocolos e detalhes de treinamento. **Questione, verifique nas
> fontes citadas e teste antes de confiar.** As afirmações sobre o código deste
> repositório foram checadas contra o próprio código; as afirmações sobre
> ferramentas/datasets externos vêm de pesquisa e devem ser reconfirmadas.

Detecção **passiva** de liveness (anti-spoofing facial) com **MiniFASNet**.
Dado um único frame RGB, classifica o rosto como **real** ou **fake** (foto
impressa, replay em tela) — sem exigir ação do usuário (piscar, virar a cabeça).

Adaptação do repositório original da Minivision para **Python 3.10+**, PyTorch
moderno e OpenCV 4.x, com backend **ONNX** adicional, execução em **webcam** em
tempo real e um conjunto de documentação de decisões em [`docs/`](docs/).

---

## Requisitos

- Python **≥ 3.10**
- Ver [`requirements.txt`](requirements.txt) (torch, opencv-python **<5.0**, onnxruntime, …)

> `opencv-python` é fixado em `<5.0` porque a versão 5 removeu o suporte a modelos
> Caffe, e o detector facial embutido é um `.caffemodel`. Ver
> [`docs/03_opencv-caffe-removal.md`](docs/03_opencv-caffe-removal.md).

---

## Instalação

```bash
cd our-liveness
bash setup.sh              # cria venv/ e instala as dependências
source venv/bin/activate
```

Ou manualmente:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Uso rápido

### Imagem única

```bash
# Amostra do repo (backend padrão: pytorch)
python predict.py --image images/sample/image_T1.jpg

# Backend ONNX (não requer torch em runtime, ~2x mais rápido em CPU)
python predict.py --image images/sample/image_T1.jpg --backend onnx

# Salvar imagem anotada (bbox + score)
python predict.py --image images/sample/image_F1.jpg --save

# Ajustar limiar de confiança (default 0.7; abaixo disso → "Inconclusive")
python predict.py --image images/sample/image_T1.jpg --threshold 0.8

# Sua própria imagem
python predict.py --image images/custom/minha_foto.jpg --save
```

### Lote (pasta inteira)

```bash
# Roda em todas as imagens de uma pasta; consolida em results/all_results.txt
python run_batch.py --input-dir images/custom --backend onnx --threshold 0.7
```

### Webcam em tempo real

```bash
python webcam.py                    # câmera 0, 15 fps, limiar 0.7
python webcam.py --camera 1 --threshold 0.8
```

Teclas: **`q`** sai · **`s`** salva o frame atual em `images/custom/`.

### Converter os modelos para ONNX

Os `.onnx` já vêm no repo, mas para regenerá-los a partir dos `.pth`:

```bash
python convert_to_onnx.py
```

---

## Como funciona (resumo)

1. **Detecção facial** — RetinaFace (Caffe) localiza o rosto e retorna a bbox.
2. **Recorte multi-escala** — dois patches são recortados em torno do rosto, cada
   um com uma escala diferente (2.7× e 4.0×) e redimensionados para 80×80.
3. **Ensemble** — dois modelos MiniFASNet (V2 e V1SE) classificam os patches; as
   probabilidades softmax são somadas.
4. **Decisão** — `argmax` sobre as classes fake/real; o score médio é comparado ao
   limiar (`--threshold`). Abaixo do limiar → "Inconclusive".

Detalhes de cada decisão de projeto estão em [`docs/`](docs/) — comece por
[`docs/README.md`](docs/README.md).

---

## Estrutura

```
our-liveness/
├── liveness/               # pacote principal
│   ├── model.py            #   arquitetura MiniFASNet (V1/V2/V1SE/V2SE)
│   ├── predictor.py        #   detector facial + inferência PyTorch
│   ├── predictor_onnx.py   #   inferência via onnxruntime
│   ├── cropper.py          #   recorte do patch de face por escala
│   └── utils.py            #   parsing de nome de modelo, kernel
├── resources/
│   ├── models/             # pesos .pth e .onnx dos dois modelos
│   └── detection/          # detector facial Caffe (RetinaFace)
├── images/
│   ├── sample/             # imagens de teste do repo original
│   └── custom/             # suas imagens (NÃO versionado — ver abaixo)
├── docs/                   # documentação de decisões (00–10)
├── predict.py              # CLI: imagem única
├── run_batch.py            # CLI: pasta inteira
├── webcam.py               # tempo real
├── convert_to_onnx.py      # exportação .pth → .onnx
├── setup.sh
└── requirements.txt
```

---

## Privacidade / dados não versionados

`images/custom/` está no `.gitignore` — **fotos de rosto são dados biométricos
pessoais** e não devem ser commitadas. Coloque suas imagens de teste lá; elas
ficam locais. Também são ignorados: `venv/`, `.env*`, imagens `*_result.*` e caches.

As imagens em `images/sample/` são as amostras públicas do repositório original.

---

## Origem e licença

Baseado em **Silent-Face-Anti-Spoofing** da [Minivision](https://www.minivision.cn/):

- Repositório: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- Commit de origem: `b6d5f04ad78778917853b25c778acef6d5626d15`
- Licença original: **Apache License 2.0**

Os pesos pré-treinados (`resources/models/*.pth`) e o detector facial
(`resources/detection/`) provêm desse repositório e mantêm a licença Apache 2.0.

> **Nota de uso:** os modelos pré-treinados foram treinados em câmeras específicas.
> Para bom desempenho na sua câmera, é recomendado retreinar — ver o plano em
> [`docs/08_plano-de-treinamento.md`](docs/08_plano-de-treinamento.md) e seguintes.

# Visão geral do projeto

## Origem

O código base é o repositório **Silent-Face-Anti-Spoofing** da Minivision (2020),
que implementa detecção passiva de vivacidade (liveness) usando a família de modelos
**MiniFASNet** — redes leves (~1.8M parâmetros) baseadas em MobileNetV2 com blocos
depthwise separáveis.

Pasta original clonada: `../Silent-Face-Anti-Spoofing/`

## O que é liveness passivo

O usuário não precisa piscar, mover a cabeça ou falar.
A rede analisa uma única imagem e classifica:

- **classe 0** → fake (foto impressa, tela, máscara)
- **classe 1** → rosto real
- **classe 2** → não usada na inferência padrão (artefato de treino)

A pontuação final é a soma das probabilidades softmax dos dois modelos ensemble.

## Estrutura de `our-liveness/`

```
our-liveness/
├── docs/                   ← você está aqui
├── liveness/               ← pacote Python (núcleo limpo)
│   ├── model.py            ← arquitetura MiniFASNet (V1, V2, V1SE, V2SE)
│   ├── predictor.py        ← FaceDetector + LivenessPredictor
│   ├── cropper.py          ← recorte do patch de face
│   └── utils.py            ← parse_model_name, get_kernel
├── resources/
│   ├── models/             ← pesos .pth pré-treinados
│   └── detection/          ← modelo Caffe para detecção facial
├── images/
│   ├── sample/             ← imagens de teste do repo original
│   └── custom/             ← coloque suas próprias imagens aqui
├── predict.py              ← ponto de entrada CLI
├── requirements.txt
└── setup.sh
```

## O que foi deixado de lado (e por quê)

| Arquivo original | Motivo |
|---|---|
| `train.py`, `train_main.py`, `train_main.py` | Treinamento fora do escopo |
| `src/data_io/dataset_folder.py` | Carregamento de datasets de treino |
| `src/data_io/dataset_loader.py` | Idem |
| `src/default_config.py` | Config de treino; usa `easydict` desnecessário |
| `saved_logs/` | Logs de treino |
| `datasets/` | Dados de treino |
| `src/data_io/transform.py` | Substituído — ver `docs/01_pixel-range-0-255.md` |
| `src/data_io/functional.py` | Idem |

## Modelos pré-treinados incluídos

| Arquivo | Arquitetura | Patch | Scale |
|---|---|---|---|
| `2.7_80x80_MiniFASNetV2.pth` | MiniFASNetV2 | 80×80 | 2.7 |
| `4_0_0_80x80_MiniFASNetV1SE.pth` | MiniFASNetV1SE | 80×80 | 4.0 |

Os dois modelos são usados em **ensemble** (soma das probabilidades softmax).

## Como reproduzir o ambiente

```bash
bash setup.sh
source venv/bin/activate
python predict.py --image images/sample/image_T1.jpg --save
```

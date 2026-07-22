# OpenCV 5.x removeu suporte a Caffe

## O problema

O detector facial usa um modelo no formato Caffe
(`Widerface-RetinaFace.caffemodel` + `deploy.prototxt`),
carregado via:

```python
cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
```

**OpenCV 5.0** (lançado em 2024) removeu os backends Caffe, Torch legado e outros
do módulo `dnn`. A função simplesmente não existe mais:

```
AttributeError: module 'cv2.dnn' has no attribute 'readNetFromCaffe'
```

## Solução aplicada

Pinamos a versão do opencv no requirements:

```
opencv-python>=4.8.0,<5.0.0
```

OpenCV 4.10 (instalado) ainda tem `readNetFromCaffe` e é compatível com Python 3.10+.

## O que isso significa a longo prazo

O modelo Caffe de detecção não vai funcionar com opencv >= 5.
Opções para o futuro:

| Alternativa | Vantagem | Desvantagem |
|---|---|---|
| Converter `.caffemodel` para ONNX | Funciona com qualquer backend | Exige conversão cuidadosa |
| Trocar por detector ONNX (ex: YuNet) | OpenCV 4+ tem `FaceDetectorYN` nativo | Formato de bbox diferente |
| Trocar por detector PyTorch (ex: MTCNN, RetinaFace-torch) | Sem dependência de Caffe | Mais uma dependência |

A migração mais direta seria usar `cv2.FaceDetectorYN` (disponível desde OpenCV 4.5.4),
que usa um modelo ONNX e tem API similar.

## Arquivo de detecção (para referência)

```
resources/detection/
├── deploy.prototxt               ← arquitetura da rede em formato Caffe
└── Widerface-RetinaFace.caffemodel ← pesos serializados no formato Caffe
```

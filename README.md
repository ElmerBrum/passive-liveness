# passive-liveness

> ⚠️ **Notice — AI-assisted content.**
> Part of this documentation (the [`docs/`](docs/) folder) and the development
> plans were produced with AI assistance. They may contain **incomplete,
> outdated, or incorrect** information — including about datasets, licenses,
> protocols, and training details. **Question it, verify against the cited
> sources, and test before relying on it.** Claims about this repository's code
> were checked against the code itself; claims about external tools/datasets come
> from research and should be reconfirmed.

Passive face **liveness** detection (anti-spoofing) with **MiniFASNet**.
Given a single RGB frame, it classifies the face as **real** or **fake** (printed
photo, screen replay) — with no user action required (no blinking or head turns).

Adaptation of Minivision's original repository for **Python 3.10+**, modern
PyTorch and OpenCV 4.x, adding an **ONNX** backend, real-time **webcam** inference,
and a set of decision docs in [`docs/`](docs/).

---

## Requirements

- Python **≥ 3.10**
- See [`requirements.txt`](requirements.txt) (torch, opencv-python **<5.0**, onnxruntime, …)

> `opencv-python` is pinned to `<5.0` because version 5 dropped Caffe model
> support, and the bundled face detector is a `.caffemodel`. See
> [`docs/03_opencv-caffe-removal.md`](docs/03_opencv-caffe-removal.md).

---

## Installation

Clone the repository and run the setup script:

```bash
git clone https://github.com/ElmerBrum/passive-liveness.git
cd passive-liveness
bash setup.sh              # creates venv/ and installs dependencies
source venv/bin/activate
```

Or set it up manually:

```bash
git clone https://github.com/ElmerBrum/passive-liveness.git
cd passive-liveness
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify the install with a sample image:

```bash
python predict.py --image images/sample/image_T1.jpg
```

---

## Quick start

### Single image

```bash
# Repo sample (default backend: pytorch)
python predict.py --image images/sample/image_T1.jpg

# ONNX backend (no torch needed at runtime, ~2x faster on CPU)
python predict.py --image images/sample/image_T1.jpg --backend onnx

# Save an annotated image (bbox + score)
python predict.py --image images/sample/image_F1.jpg --save

# Adjust the confidence threshold (default 0.7; below it → "Inconclusive")
python predict.py --image images/sample/image_T1.jpg --threshold 0.8

# Your own image
python predict.py --image images/custom/my_photo.jpg --save
```

### Batch (whole folder)

```bash
# Runs on every image in a folder; consolidates into results/all_results.txt
python run_batch.py --input-dir images/custom --backend onnx --threshold 0.7
```

### Real-time webcam

```bash
python webcam.py                    # camera 0, 15 fps, threshold 0.7
python webcam.py --camera 1 --threshold 0.8
```

Keys: **`q`** quit · **`s`** save the current frame to `images/custom/`.

### Convert the models to ONNX

The `.onnx` files ship with the repo, but to regenerate them from the `.pth`:

```bash
python convert_to_onnx.py
```

---

## How it works (overview)

1. **Face detection** — RetinaFace (Caffe) locates the face and returns a bbox.
2. **Multi-scale cropping** — two patches are cropped around the face, each at a
   different scale (2.7× and 4.0×) and resized to 80×80.
3. **Ensemble** — two MiniFASNet models (V2 and V1SE) classify the patches; their
   softmax probabilities are summed.
4. **Decision** — `argmax` over the fake/real classes; the mean score is compared
   to the threshold (`--threshold`). Below it → "Inconclusive".

Details of each design decision are in [`docs/`](docs/) — start with
[`docs/README.md`](docs/README.md).

---

## Project structure

```
passive-liveness/
├── liveness/               # core package
│   ├── model.py            #   MiniFASNet architecture (V1/V2/V1SE/V2SE)
│   ├── predictor.py        #   face detector + PyTorch inference
│   ├── predictor_onnx.py   #   onnxruntime inference
│   ├── cropper.py          #   per-scale face patch cropping
│   └── utils.py            #   model-name parsing, kernel
├── resources/
│   ├── models/             # .pth and .onnx weights for both models
│   └── detection/          # Caffe face detector (RetinaFace)
├── images/
│   ├── sample/             # test images from the original repo
│   └── custom/             # your images (NOT versioned — see below)
├── docs/                   # decision docs (00–10)
├── predict.py              # CLI: single image
├── run_batch.py            # CLI: whole folder
├── webcam.py               # real-time
├── convert_to_onnx.py      # .pth → .onnx export
├── setup.sh
└── requirements.txt
```

---

## Privacy / untracked data

`images/custom/` is in `.gitignore` — **face photos are personal biometric data**
and must not be committed. Put your test images there; they stay local. Also
ignored: `venv/`, `.env*`, `*_result.*` images, and caches.

The images under `images/sample/` are the public samples from the original repo.

---

## Origin and license

Based on **Silent-Face-Anti-Spoofing** by [Minivision](https://www.minivision.cn/):

- Repository: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
- Source commit: `b6d5f04ad78778917853b25c778acef6d5626d15`
- Original license: **Apache License 2.0**

The pre-trained weights (`resources/models/*.pth`) and the face detector
(`resources/detection/`) come from that repository and keep the Apache 2.0 license.

> **Usage note:** the pre-trained models were trained on specific cameras. For good
> performance on your own camera, retraining is recommended — see the plan in
> [`docs/08_plano-de-treinamento.md`](docs/08_plano-de-treinamento.md) onward.

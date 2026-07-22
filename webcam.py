#!/usr/bin/env python3
"""
Liveness passivo em tempo real via webcam.

Uso:
    python webcam.py
    python webcam.py --camera 1        # outra câmera
    python webcam.py --threshold 0.8
    python webcam.py --target-fps 15   # padrão

Teclas durante a execução:
    q      encerra
    s      salva o frame atual em images/custom/

Arquitetura:
    Thread principal  → captura frames e exibe (sempre a ~target-fps)
    Thread de worker  → roda detecção + liveness no frame mais recente
    Resultado         → compartilhado via lock; display usa o último válido

    Isso mantém a janela fluida mesmo quando a inferência demora mais
    que o budget de um frame.
"""

import argparse
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT  = Path(__file__).parent
MODELS_DIR    = PROJECT_ROOT / "resources" / "models"
DETECTION_DIR = PROJECT_ROOT / "resources" / "detection"

# Cores BGR
COLOR_REAL        = (0, 200, 0)    # verde
COLOR_FAKE        = (0, 0, 220)    # vermelho
COLOR_INCONCLUSIVE = (0, 180, 255) # amarelo
COLOR_NO_FACE     = (120, 120, 120)


def color_for(label: int) -> tuple:
    return {1: COLOR_REAL, 0: COLOR_FAKE}.get(label, COLOR_INCONCLUSIVE)


def verdict_for(label: int) -> str:
    return {1: "Real", 0: "Fake"}.get(label, "Inconclusive")


class InferenceWorker:
    """
    Roda detecção + liveness em background.
    O display thread lê .result sem bloquear.
    """

    def __init__(self, threshold: float):
        from liveness.predictor_onnx import LivenessPredictorONNX
        from liveness.cropper import CropImage

        self._predictor = LivenessPredictorONNX(
            models_dir=MODELS_DIR,
            detection_model_dir=DETECTION_DIR,
        )
        self._cropper  = CropImage()
        self._threshold = threshold

        # Estado compartilhado
        self._lock         = threading.Lock()
        self._latest_frame = None   # frame para processar
        self._frame_ready  = threading.Event()
        self.result        = None   # último resultado válido
        self.inference_ms  = 0.0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Envia um frame para processar. Se o worker ainda está ocupado, descarta o anterior."""
        with self._lock:
            self._latest_frame = frame.copy()
        self._frame_ready.set()

    def _loop(self) -> None:
        while True:
            self._frame_ready.wait()
            self._frame_ready.clear()

            with self._lock:
                frame = self._latest_frame

            if frame is None:
                continue

            try:
                t0 = time.perf_counter()
                label, score, bbox, per_model = self._predictor.predict_ensemble(
                    frame, self._cropper, threshold=self._threshold
                )
                elapsed_ms = (time.perf_counter() - t0) * 1000

                with self._lock:
                    self.result       = (label, score, bbox, per_model)
                    self.inference_ms = elapsed_ms

            except Exception:
                # Face não detectada ou erro de runtime — mantém último resultado
                pass

    def get_result(self):
        with self._lock:
            return self.result, self.inference_ms


def draw_overlay(frame: np.ndarray, result, inference_ms: float,
                 display_fps: float, threshold: float) -> np.ndarray:
    h, w = frame.shape[:2]
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = w / 1280  # escala relativa à largura

    if result is None:
        cv2.putText(frame, "Aguardando deteccao...", (10, 30),
                    font, font_scale, COLOR_NO_FACE, 1, cv2.LINE_AA)
        return frame

    label, score, bbox, per_model = result
    color   = color_for(label)
    verdict = verdict_for(label)

    # Bounding box
    x, y, bw, bh = bbox
    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

    # Verdict + score acima do bbox
    tag = f"{verdict}  {score:.2f}"
    cv2.putText(frame, tag, (x, max(y - 8, 20)),
                font, font_scale * 1.1, color, 2, cv2.LINE_AA)

    # Painel de info no canto superior esquerdo
    lines = [
        f"FPS: {display_fps:.1f}   Inf: {inference_ms:.0f}ms",
        f"Limiar: {threshold}",
    ]
    if per_model:
        for m in per_model:
            lines.append(f"{m['name'].split('_')[-1]}  "
                         f"fake={m['fake']:.3f}  real={m['real']:.3f}")

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (8, 22 + i * int(22 * font_scale * 1.4)),
                    font, font_scale * 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def run(camera_idx: int, target_fps: int, threshold: float) -> None:
    cap = cv2.VideoCapture(camera_idx)
    if not cap.isOpened():
        sys.exit(f"ERRO: não foi possível abrir a câmera {camera_idx}.")

    # 640x480 é o modo landscape padrão suportado pela maioria das webcams.
    # Depois recortamos para 3:4 (360x480) que é a proporção do treino do modelo.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Lê frames de teste para confirmar que a câmera está funcionando
    for _ in range(5):
        ret, test = cap.read()
        if ret and test is not None and test.size > 0:
            break
    else:
        cap.release()
        sys.exit("ERRO: câmera abriu mas não retornou frames válidos.")

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Camera {camera_idx}: {actual_w}x{actual_h}")
    print(f"Target: {target_fps}fps  |  Limiar: {threshold}")
    print("Pressione 'q' para sair, 's' para salvar frame.\n")

    worker      = InferenceWorker(threshold)
    frame_ms    = 1000 / target_fps
    last_submit = 0.0

    # Métricas de FPS do display
    fps_counter = 0
    fps_ts      = time.perf_counter()
    display_fps = 0.0

    save_dir = PROJECT_ROOT / "images" / "custom"

    # Nome ASCII puro — caracteres Unicode no título da janela quebram
    # alguns backends do OpenCV (GTK, Qt) em Linux
    WIN = "Liveness MiniFASNet  |  q=sair  s=salvar"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, actual_w, actual_h)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            continue

        now = time.perf_counter()

        # Envia frame para o worker na frequência alvo
        if (now - last_submit) * 1000 >= frame_ms:
            worker.submit(frame)
            last_submit = now

        result, inference_ms = worker.get_result()
        display = draw_overlay(frame.copy(), result, inference_ms, display_fps, threshold)
        cv2.imshow(WIN, display)

        # FPS do display
        fps_counter += 1
        if now - fps_ts >= 1.0:
            display_fps = fps_counter / (now - fps_ts)
            fps_counter = 0
            fps_ts      = now

        # waitKey consume a fila de eventos do backend gráfico.
        # Usamos o tempo restante do budget do frame para não gastar CPU à toa.
        elapsed_ms = (time.perf_counter() - now) * 1000
        wait_ms    = max(1, int(frame_ms - elapsed_ms))
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord('q'):
            break
        if key == ord('s'):
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"webcam_{ts}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"Frame salvo -> {path}")

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Liveness em tempo real — webcam")
    parser.add_argument("--camera",     type=int,   default=0,
                        help="Índice da câmera (default: 0)")
    parser.add_argument("--target-fps", type=int,   default=15,
                        help="FPS alvo (default: 15)")
    parser.add_argument("--threshold",  type=float, default=0.7,
                        help="Score mínimo para aceitar decisão (default: 0.7)")
    args = parser.parse_args()
    run(args.camera, args.target_fps, args.threshold)


if __name__ == "__main__":
    main()

import os
import math


def get_kernel(height: int, width: int) -> tuple[int, int]:
    """Compute the conv6_dw kernel size from the input patch dimensions."""
    return ((height + 15) // 16, (width + 15) // 16)


def parse_model_name(filename: str) -> tuple[int, int, str, float | None]:
    """
    Extract (h_input, w_input, model_type, scale) from a model filename.

    Naming convention used by the Minivision pre-trained weights:
      <scale>_<h>x<w>_<ModelType>.pth         e.g. 2.7_80x80_MiniFASNetV2.pth
      <a>_<b>_<c>_<h>x<w>_<ModelType>.pth    e.g. 4_0_0_80x80_MiniFASNetV1SE.pth

    Logic (mirrors the original repo):
      - Split by '_' and drop the last token (<ModelType>.pth).
      - The last remaining token is "<h>x<w>".
      - The very first token is the scale value (or "org" → None).
      Note: "4_0_0" encodes scale 4.0; only info[0] = "4" is used.
    """
    parts = filename.split('_')
    info  = parts[:-1]                             # drop "MiniFASNetXxx.pth"
    model_type = os.path.splitext(parts[-1])[0]   # strip .pth from last token
    h_input, w_input = info[-1].split('x')        # last remaining is "80x80"
    scale = None if info[0] == 'org' else float(info[0])
    return int(h_input), int(w_input), model_type, scale

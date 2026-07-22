import cv2
import numpy as np


class CropImage:
    """Crop a face patch from an image using a detected bounding box."""

    @staticmethod
    def _get_new_box(src_w: int, src_h: int, bbox: list, scale: float) -> tuple:
        x, y, box_w, box_h = bbox
        scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
        new_w, new_h = box_w * scale, box_h * scale
        cx, cy = x + box_w / 2, y + box_h / 2

        x0, y0 = cx - new_w / 2, cy - new_h / 2
        x1, y1 = cx + new_w / 2, cy + new_h / 2

        if x0 < 0:
            x1 -= x0; x0 = 0
        if y0 < 0:
            y1 -= y0; y0 = 0
        if x1 > src_w - 1:
            x0 -= x1 - src_w + 1; x1 = src_w - 1
        if y1 > src_h - 1:
            y0 -= y1 - src_h + 1; y1 = src_h - 1

        return int(x0), int(y0), int(x1), int(y1)

    def crop(self, org_img: np.ndarray, bbox: list, scale: float,
             out_w: int, out_h: int, crop: bool = True) -> np.ndarray:
        if not crop:
            return cv2.resize(org_img, (out_w, out_h))

        src_h, src_w = org_img.shape[:2]
        x0, y0, x1, y1 = self._get_new_box(src_w, src_h, bbox, scale)
        patch = org_img[y0: y1 + 1, x0: x1 + 1]
        return cv2.resize(patch, (out_w, out_h))

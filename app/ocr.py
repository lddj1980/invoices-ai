"""Utility functions for performing OCR on images using OpenCV and Tesseract.

This module implements a lightweight version of the table-extraction pipeline
shared in the user request. It focuses on robust pre-processing to improve OCR
results when invoices are provided as images (PNG/JPEG).

The functions are imported lazily so that missing optional dependencies (such as
``opencv-python`` or ``pytesseract``) do not break module import. When these
libraries are not available the caller will receive an ``ImportError``.
"""
from __future__ import annotations

from typing import List, Tuple

# Configuration defaults mirroring the provided snippet
LANG = "por+eng"
PSM = 6
OEM = 3
MIN_CELL_W, MIN_CELL_H = 20, 15
BINARY_BLOCK_SIZE = 31
BINARY_C = 9


def _load_cv2_modules():  # pragma: no cover - thin wrapper for lazy imports
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        from PIL import Image  # type: ignore
        import pytesseract  # type: ignore
    except Exception as exc:  # pragma: no cover - handled at runtime
        raise ImportError("OpenCV and Tesseract dependencies are required") from exc
    return cv2, np, pd, Image, pytesseract


def load_and_preprocess_bytes(data: bytes):
    """Decode image bytes and apply basic normalisation, deskew and thresholding.

    Parameters
    ----------
    data:
        Raw image bytes (PNG/JPEG).
    """
    cv2, np, pd_unused, Image_unused, pytesseract_unused = _load_cv2_modules()

    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    coords = np.column_stack(np.where(gray < 255))
    angle = 0.0
    if len(coords) > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        BINARY_BLOCK_SIZE,
        BINARY_C,
    )
    return img, gray, bw


def get_table_mask(bw):
    cv2, np_unused, pd_unused, Image_unused, pytesseract_unused = _load_cv2_modules()

    horizontal = bw.copy()
    cols = horizontal.shape[1]
    horiz_size = max(10, cols // 40)
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horiz_size, 1))
    horizontal = cv2.erode(horizontal, horiz_kernel)
    horizontal = cv2.dilate(horizontal, horiz_kernel)

    vertical = bw.copy()
    rows = vertical.shape[0]
    vert_size = max(10, rows // 40)
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_size))
    vertical = cv2.erode(vertical, vert_kernel)
    vertical = cv2.dilate(vertical, vert_kernel)

    mask = cv2.add(horizontal, vertical)
    return mask


def find_cells(mask) -> List[Tuple[int, int, int, int]]:
    cv2, np, pd_unused, Image_unused, pytesseract_unused = _load_cv2_modules()

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= MIN_CELL_W and h >= MIN_CELL_H:
            boxes.append((x, y, w, h))
    if boxes:
        areas = [w * h for (_, _, w, h) in boxes]
        cut = sorted(areas)[int(len(areas) * 0.95)]
        boxes = [b for b, a in zip(boxes, areas) if a <= cut or len(boxes) < 50]
    return boxes


def sort_boxes_into_grid(boxes: List[Tuple[int, int, int, int]], y_tol: int = 10, x_tol: int = 10):
    boxes_sorted = sorted(boxes, key=lambda b: (b[1], b[0]))

    rows = []
    current_row = []
    for b in boxes_sorted:
        if not current_row:
            current_row.append(b)
            continue
        if abs(b[1] - current_row[0][1]) <= y_tol:
            current_row.append(b)
        else:
            rows.append(sorted(current_row, key=lambda bb: bb[0]))
            current_row = [b]
    if current_row:
        rows.append(sorted(current_row, key=lambda bb: bb[0]))

    max_cols = max((len(r) for r in rows), default=0)
    return rows, max_cols


def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\n", " ").replace("\t", " ")
    t = " ".join(t.split())
    return t


def ocr_cell(img, box) -> str:
    cv2, np_unused, pd_unused, Image, pytesseract = _load_cv2_modules()

    x, y, w, h = box
    roi = img[y : y + h, x : x + w]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_bw = cv2.adaptiveThreshold(
        roi_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    pil = Image.fromarray(roi_bw)
    text = pytesseract.image_to_string(pil, lang=LANG, config=f"--oem {OEM} --psm {PSM}")
    return clean_text(text)


def build_dataframe(img, rows, max_cols):
    import pandas as pd  # local import to keep dependency optional

    data = []
    for r in rows:
        row_text = [ocr_cell(img, b) for b in r]
        while len(row_text) < max_cols:
            row_text.append("")
        data.append(row_text)

    header = data[0] if data else []
    looks_header = any(
        h.lower()
        in [
            "no.",
            "description",
            "size",
            "qty",
            "n.w.",
            "g.w.",
            "volume",
            "pcs/ctn",
            "ctn/plt",
            "plts",
        ]
        for h in header
    )
    if looks_header:
        df = pd.DataFrame(data[1:], columns=header)
    else:
        cols = [f"col_{i+1}" for i in range(max_cols)]
        df = pd.DataFrame(data, columns=cols)
    return df


def image_to_text(data: bytes) -> str:
    """Extract textual content from an invoice image.

    The function performs table detection and OCR on individual cells. The
    detected cells are concatenated row-wise to form a raw text representation
    of the table, which works well with the rest of the pipeline that expects
    plain text.
    """
    _, _, pd_unused, Image_unused, pytesseract_unused = _load_cv2_modules()

    img, gray, bw = load_and_preprocess_bytes(data)
    mask = get_table_mask(bw)
    boxes = find_cells(mask)

    if boxes:
        rows, max_cols = sort_boxes_into_grid(boxes, y_tol=12, x_tol=12)
        df = build_dataframe(img, rows, max_cols)
        text = "\n".join(" ".join(row) for row in df.values.tolist())
        text = clean_text(text)
        if text:
            return text

    # Fallback: run OCR on the whole image
    cv2, np_unused, pd_unused, Image, pytesseract = _load_cv2_modules()
    pil = Image.fromarray(bw)
    text = pytesseract.image_to_string(pil, lang=LANG, config=f"--oem {OEM} --psm {PSM}")
    return clean_text(text)

__all__ = ["image_to_text"]


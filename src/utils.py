import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("ChangeDetector")


def json_serializer(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def next_run_dir(base: str) -> Path:
    base_path = Path(base)
    base_path.mkdir(parents=True, exist_ok=True)
    existing = [
        int(p.name) for p in base_path.iterdir()
        if p.is_dir() and p.name.isdigit()
    ]
    run_dir = base_path / str(max(existing, default=0) + 1)
    run_dir.mkdir(parents=True)
    logger.info(f"Run directory: {run_dir}")
    return run_dir


def load_rois_from_file(roi_file: str) -> Optional[list[tuple[int, int, int, int]]]:
    """
    Reads roi.json and returns the ROI list in (x1, y1, x2, y2) format.
    File format:
      { "rois": [ {"id":1, "x":..., "y":..., "width":..., "height":...}, ... ] }
    Returns None if the file does not exist or an error occurs.
    """
    roi_path = Path(roi_file)
    if not roi_path.exists():
        logger.info(f"ROI file not found: {roi_path} — Using GUI.")
        return None

    try:
        with open(roi_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_rois = data.get("rois", [])
        if not raw_rois:
            logger.warning(f"ROI file is empty: {roi_path} — Using GUI.")
            return None

        rois: list[tuple[int, int, int, int]] = []
        for entry in raw_rois:
            roi_id = entry.get("id", "?")
            x = int(entry["x"])
            y = int(entry["y"])
            x2 = x + int(entry["width"])
            y2 = y + int(entry["height"])
            rois.append((x, y, x2, y2))
            logger.info(
                f"ROI #{roi_id} loaded → "
                f"({x},{y})-({x2},{y2})  "
                f"[{int(entry['width'])}×{int(entry['height'])}]"
            )

        logger.info(
            f"ROI file loaded: {roi_path}  |  "
            f"ROI count: {len(rois)}  |  "
            f"Original reference image: {data.get('image_path', 'unknown')}"
        )
        return rois

    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        logger.error(f"Error reading ROI file ({roi_path}): {exc} — Using GUI.")
        return None


class ROISelector:
    _RECT_COLOR = (0, 255, 0)
    _RECT_COLOR_DONE = (0, 200, 255)
    _THICKNESS = 2

    def __init__(self, image: np.ndarray, window_name: str = "Select ROIs"):
        self._orig = image.copy()
        self._canvas = image.copy()
        self._win = window_name
        self._rois: list[tuple[int, int, int, int]] = []
        self._drawing = False
        self._start = (0, 0)
        self._current = (0, 0)

    def _mouse_cb(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._start = (x, y)
            self._current = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._current = (x, y)
            self._redraw(live_rect=(self._start, self._current))
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1, y1 = min(self._start[0], x), min(self._start[1], y)
            x2, y2 = max(self._start[0], x), max(self._start[1], y)
            if x2 - x1 > 5 and y2 - y1 > 5:
                self._rois.append((x1, y1, x2, y2))
                logger.info(f"ROI #{len(self._rois)} → ({x1},{y1})-({x2},{y2})")
            self._redraw()

    def _redraw(self, live_rect=None):
        self._canvas = self._orig.copy()
        for i, (x1, y1, x2, y2) in enumerate(self._rois):
            cv2.rectangle(self._canvas, (x1, y1), (x2, y2),
                          self._RECT_COLOR_DONE, self._THICKNESS)
            cv2.putText(self._canvas, f"ROI {i+1}", (x1 + 4, y1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, self._RECT_COLOR_DONE,
                        2, cv2.LINE_AA)
        if live_rect:
            cv2.rectangle(self._canvas, live_rect[0], live_rect[1],
                          self._RECT_COLOR, self._THICKNESS)
        self._draw_help()
        cv2.imshow(self._win, self._canvas)

    def _draw_help(self):
        h = self._canvas.shape[0]
        for txt, pos in [("Drag: draw ROI", (10, h - 40)),
                         ("R: reset  |  Q: confirm", (10, h - 15))]:
            cv2.putText(self._canvas, txt, pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    def run(self) -> list[tuple[int, int, int, int]]:
        cv2.namedWindow(self._win, cv2.WINDOW_NORMAL)
        cv2.waitKey(1)
        cv2.setMouseCallback(self._win, self._mouse_cb)
        self._redraw()
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                if not self._rois:
                    logger.warning("No ROI has been selected.")
                    continue
                break
            elif key == ord("r"):
                self._rois.clear()
                self._redraw()
        cv2.destroyWindow(self._win)
        return list(self._rois)


def draw_box(img: np.ndarray, box: np.ndarray, color: tuple, label: str):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 2, y1), color, -1)
    cv2.putText(img, label, (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

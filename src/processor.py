import logging
import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO

from .config import DetectorConfig, default_office_classes
from .utils import (
    ROISelector,
    load_rois_from_file,
    next_run_dir,
    json_serializer,
    draw_box,
    ensure_dir,
)

logger = logging.getLogger("ChangeDetector")


@dataclass
class DetectedObject:
    box: np.ndarray
    class_id: int
    class_name: str
    confidence: float
    embedding: Optional[np.ndarray] = None

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2,
                         (self.box[1] + self.box[3]) / 2])

    @property
    def area(self) -> float:
        return (self.box[2] - self.box[0]) * (self.box[3] - self.box[1])


class ModelManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def initialize(self, model_name: str):
        if self._initialized:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device: {self.device}")
        t0 = time.perf_counter()

        self.yolo = YOLO(model_name)
        self.yolo.to(self.device)  # [GPU-FIX] Explicitly move YOLO to GPU
        logger.info(f"YOLO model loaded: {model_name} in {time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        self.dino = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False, force_reload=False
        ).to(self.device).eval()
        logger.info(f"DINOv2 loaded in {time.perf_counter()-t0:.2f}s")
        self._initialized = True


class ChangeDetector:
    _COLOR_MATCHED = (0, 200, 0)
    _COLOR_REMOVED = (0, 0, 220)
    _COLOR_ADDED = (220, 0, 0)
    _COLOR_RESCUED = (0, 180, 255)

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self.models = ModelManager()
        self.models.initialize(model_name=self.cfg.yolo_model)

        self._reference_cache = {}
        self._is_world_model = hasattr(self.models.yolo, "set_classes")

        if self._is_world_model:
            self._active_classes = (
                self.cfg.target_classes if self.cfg.target_classes
                else default_office_classes()
            )
            self.models.yolo.set_classes(self._active_classes)
            logger.info(f"YOLO-World classes set: {self._active_classes}")
        else:
            self._active_classes = []
            logger.info("Standard YOLO model detected — using built-in COCO classes.")

        # Ensure output dir exists in project root (as requested)
        ensure_dir(self.cfg.base_output_dir)

    # ── Preprocessing ─────────────────────────────────────────────────────────
    def _crop_with_padding(self, img: np.ndarray, box: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        pad = self.cfg.crop_padding
        x1, y1 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        x2, y2 = min(w, int(box[2]) + pad), min(h, int(box[3]) + pad)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.cfg.dino_input_size, self.cfg.dino_input_size, 3),
                            dtype=np.uint8)
        return crop

    def _preprocess_crop(self, crop: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_bgr = cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)
        src_h, src_w = img_bgr.shape[:2]
        target = self.cfg.dino_input_size
        interp = cv2.INTER_CUBIC if (src_h < target or src_w < target) else cv2.INTER_AREA
        img = cv2.resize(img_bgr, (target, target), interpolation=interp)
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        return ((img - mean) / std).transpose(2, 0, 1)

    # ── Detection ─────────────────────────────────────────────────────────────
    def _detect(self, img: np.ndarray, label: str) -> list[DetectedObject]:
        t0 = time.perf_counter()
        results = self.models.yolo.predict(
            img,
            conf=self.cfg.yolo_conf,
            iou=self.cfg.yolo_iou,
            device=self.models.device,
            verbose=False,
        )
        boxes = results[0].boxes
        objects = []
        for box in boxes:
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id]
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].cpu().numpy()
            if cls_name.lower() in self.cfg.excluded_classes:
                continue
            objects.append(DetectedObject(
                box=xyxy, class_id=cls_id, class_name=cls_name, confidence=conf
            ))
        logger.info(
            f"[{label}] {len(objects)} objects "
            f"({len(boxes) - len(objects)} excluded) "
            f"in {time.perf_counter()-t0:.3f}s"
        )
        return objects

    # ── Embedding ─────────────────────────────────────────────────────────────
    def _embed(self, img: np.ndarray, objects: list[DetectedObject], label: str):
        if not objects:
            return
        t0 = time.perf_counter()
        processed = [self._preprocess_crop(self._crop_with_padding(img, o.box))
                     for o in objects]
        all_embs = []
        for i in range(0, len(processed), self.cfg.batch_size):
            batch = np.array(processed[i: i + self.cfg.batch_size])
            tensor = torch.tensor(batch, dtype=torch.float32).to(self.models.device)
            with torch.no_grad():
                embs = self.models.dino(tensor)
            all_embs.append(torch.nn.functional.normalize(embs, dim=1).cpu().numpy())
        all_embs_np = np.concatenate(all_embs, axis=0)
        for obj, emb in zip(objects, all_embs_np):
            obj.embedding = emb
        logger.info(f"[{label}] Embedded {len(objects)} in {time.perf_counter()-t0:.3f}s")

    # ── Matching ──────────────────────────────────────────────────────────────
    def _position_similarity(
        self,
        ref_objs: list[DetectedObject],
        tgt_objs: list[DetectedObject],
        img_diag: float,
    ) -> np.ndarray:
        ref_c = np.array([o.center for o in ref_objs])
        tgt_c = np.array([o.center for o in tgt_objs])
        dists = np.linalg.norm(
            tgt_c[:, np.newaxis, :] - ref_c[np.newaxis, :, :], axis=2
        ) / img_diag
        tol_norm = self.cfg.position_tolerance_px / img_diag
        return np.exp(-dists / (tol_norm + 1e-6))

    def _build_cost_matrix(
        self,
        ref_objs: list[DetectedObject],
        tgt_objs: list[DetectedObject],
        img_diag: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref_embs = np.array([o.embedding for o in ref_objs])
        tgt_embs = np.array([o.embedding for o in tgt_objs])
        visual_sim = np.clip(cosine_similarity(tgt_embs, ref_embs), 0, 1)
        pos_sim = self._position_similarity(ref_objs, tgt_objs, img_diag)
        combined = (1 - self.cfg.alpha) * visual_sim + self.cfg.alpha * pos_sim
        return 1.0 - combined, combined

    def _match(
        self,
        ref_objs: list[DetectedObject],
        tgt_objs: list[DetectedObject],
        img_diag: float,
    ) -> tuple[list, list, list]:
        if not ref_objs:
            return [], [], list(range(len(tgt_objs)))
        if not tgt_objs:
            return [], list(range(len(ref_objs))), []

        cost_matrix, score_matrix = self._build_cost_matrix(ref_objs, tgt_objs, img_diag)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs = []
        unmatched_ref = set(range(len(ref_objs)))
        unmatched_tgt = set(range(len(tgt_objs)))

        for tgt_idx, ref_idx in zip(row_ind, col_ind):
            score = score_matrix[tgt_idx, ref_idx]
            if score >= self.cfg.similarity_threshold:
                matched_pairs.append((tgt_idx, ref_idx, score))
                unmatched_ref.discard(ref_idx)
                unmatched_tgt.discard(tgt_idx)

        logger.info(
            f"Matching → matched={len(matched_pairs)}, "
            f"removed={len(unmatched_ref)}, added={len(unmatched_tgt)}"
        )
        return matched_pairs, list(unmatched_ref), list(unmatched_tgt)

    # ── Double-Check ──────────────────────────────────────────────────────────
    def _double_check(self, candidate_objs, other_img, label):
        if not candidate_objs:
            return [], []

        processed = [
            self._preprocess_crop(self._crop_with_padding(other_img, obj.box))
            for obj in candidate_objs
        ]

        all_embs = []
        for i in range(0, len(processed), self.cfg.batch_size):
            batch = np.array(processed[i:i + self.cfg.batch_size])
            tensor = torch.tensor(batch, dtype=torch.float32).to(self.models.device)
            with torch.no_grad():
                embs = self.models.dino(tensor)
            all_embs.append(torch.nn.functional.normalize(embs, dim=1).cpu().numpy())
        other_embs = np.concatenate(all_embs, axis=0)

        confirmed, rescued = [], []
        for i, (obj, other_emb) in enumerate(zip(candidate_objs, other_embs)):
            sim = float(cosine_similarity(
                obj.embedding[np.newaxis], other_emb[np.newaxis]
            )[0, 0])
            logger.debug(f"[double-check/{label}] {obj.class_name} sim={sim:.3f}")
            if sim >= self.cfg.double_check_threshold:
                rescued.append(i)
                logger.info(f"[double-check/{label}] RESCUED {obj.class_name} (sim={sim:.3f})")
            else:
                confirmed.append(i)
        return confirmed, rescued

    # ── Visualization ─────────────────────────────────────────────────────────
    def _visualize(
        self,
        ref_img: np.ndarray,
        tgt_img: np.ndarray,
        ref_objs: list[DetectedObject],
        tgt_objs: list[DetectedObject],
        matched_pairs: list,
        unmatched_ref: list,
        unmatched_tgt: list,
        rescued_ref: list,
        rescued_tgt: list,
        output_path: str,
    ):
        ref_vis = ref_img.copy()
        tgt_vis = tgt_img.copy()

        matched_ref_set = {r for _, r, _ in matched_pairs}
        matched_tgt_set = {t for t, _, _ in matched_pairs}
        rescued_ref_set = set(rescued_ref)
        rescued_tgt_set = set(rescued_tgt)

        for i, obj in enumerate(ref_objs):
            if i in matched_ref_set:
                draw_box(ref_vis, obj.box, self._COLOR_MATCHED, f"{obj.class_name} OK")
            elif i in rescued_ref_set:
                draw_box(ref_vis, obj.box, self._COLOR_RESCUED, f"{obj.class_name} OK*")
            else:
                draw_box(ref_vis, obj.box, self._COLOR_REMOVED, f"{obj.class_name} REMOVED")

        for i, obj in enumerate(tgt_objs):
            if i in matched_tgt_set:
                draw_box(tgt_vis, obj.box, self._COLOR_MATCHED, f"{obj.class_name} OK")
            elif i in rescued_tgt_set:
                draw_box(tgt_vis, obj.box, self._COLOR_RESCUED, f"{obj.class_name} OK*")
            else:
                draw_box(tgt_vis, obj.box, self._COLOR_ADDED, f"{obj.class_name} ADDED")

        for tgt_idx, ref_idx, score in matched_pairs:
            tc = tgt_objs[tgt_idx].center.astype(int)
            cv2.putText(tgt_vis, f"{score:.2f}", tuple(tc),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self._COLOR_MATCHED, 1, cv2.LINE_AA)

        h = max(ref_vis.shape[0], tgt_vis.shape[0])
        ref_vis = cv2.copyMakeBorder(ref_vis, 0, h - ref_vis.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        tgt_vis = cv2.copyMakeBorder(tgt_vis, 0, h - tgt_vis.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        combined = np.hstack([ref_vis, tgt_vis])

        for text, x in [("REFERENCE", 10), ("TARGET", ref_vis.shape[1] + 10)]:
            cv2.putText(combined, text, (x, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        legend_y = combined.shape[0] - 10
        for color, txt, offset in [
            (self._COLOR_MATCHED, "MATCHED", 10),
            (self._COLOR_REMOVED, "REMOVED", 130),
            (self._COLOR_ADDED, "ADDED", 250),
            (self._COLOR_RESCUED, "OK*", 360),
        ]:
            cv2.rectangle(combined, (offset, legend_y - 15),
                          (offset + 15, legend_y), color, -1)
            cv2.putText(combined, txt, (offset + 20, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imwrite(output_path, combined)
        logger.info(f"Visualization saved → {output_path}")

    # ── Reference Cache ───────────────────────────────────────────────────────
    def _reference_cache_key(self, ref_path: str, roi: tuple[int, int, int, int]):
        path = Path(ref_path)
        try:
            stat = path.stat()
            file_signature = (str(path.resolve()), int(stat.st_mtime_ns), int(stat.st_size))
        except OSError:
            file_signature = (str(path.resolve()), None, None)

        classes_signature = tuple(self._active_classes) if self._is_world_model else None

        return (
            file_signature,
            tuple(int(v) for v in roi),
            self.cfg.yolo_model,
            classes_signature,
            float(self.cfg.yolo_conf),
            float(self.cfg.yolo_iou),
            tuple(sorted(self.cfg.excluded_classes)),
        )

    def _prepare_reference_cache(
        self,
        ref_img_full: np.ndarray,
        rois: list[tuple[int, int, int, int]],
        ref_path: str,
    ) -> dict:
        cache_start = time.perf_counter()
        reference_data = {}
        cache_hits = 0
        cache_misses = 0

        for roi_idx, roi in enumerate(rois):
            key = self._reference_cache_key(ref_path, roi)
            cached = self._reference_cache.get(key)

            if cached is not None:
                reference_data[roi_idx] = cached
                cache_hits += 1
                logger.info(
                    f"[CACHE] ROI #{roi_idx + 1}: reference objects/embeddings reused "
                    f"({len(cached['ref_objs'])} objects)"
                )
                continue

            x1, y1, x2, y2 = roi
            ref_patch = ref_img_full[y1:y2, x1:x2]
            if ref_patch.size == 0:
                raise ValueError(f"Reference ROI #{roi_idx + 1} is empty: {roi}")

            logger.info(f"[CACHE] Preparing reference ROI #{roi_idx + 1}: {roi}")
            ref_objs = self._detect(ref_patch, "REF/CACHE")
            self._embed(ref_patch, ref_objs, "REF/CACHE")

            cached = {"roi": tuple(roi), "ref_objs": ref_objs, "ref_shape": tuple(ref_patch.shape[:2])}
            self._reference_cache[key] = cached
            reference_data[roi_idx] = cached
            cache_misses += 1

            logger.info(f"[CACHE] ROI #{roi_idx + 1}: reference cached ({len(ref_objs)} objects)")

        cache_elapsed = time.perf_counter() - cache_start
        logger.info(
            f"Reference cache ready | hits={cache_hits} misses={cache_misses} "
            f"| cache_time={cache_elapsed:.3f}s | EXCLUDED from measured time"
        )
        return reference_data

    # ── Single ROI Compare ────────────────────────────────────────────────────
    def _compare_roi(
        self,
        ref_img_full: np.ndarray,
        tgt_img_full: np.ndarray,
        roi: tuple[int, int, int, int],
        reference_cache: dict,
        roi_output_dir: Path,
        ref_path: str,
        tgt_path: str,
    ) -> dict:
        x1, y1, x2, y2 = roi
        ref_patch = ref_img_full[y1:y2, x1:x2]
        tgt_patch = tgt_img_full[y1:y2, x1:x2]

        cv2.imwrite(str(roi_output_dir / "ref_patch.jpg"), ref_patch)
        cv2.imwrite(str(roi_output_dir / "tgt_patch.jpg"), tgt_patch)

        h, w = ref_patch.shape[:2]
        img_diag = float(np.sqrt(h**2 + w**2))

        ref_objs = reference_cache["ref_objs"]
        logger.info(f"[REF/CACHE] Reusing {len(ref_objs)} objects/embeddings for ROI {roi}")

        tgt_objs = self._detect(tgt_patch, "TGT")
        self._embed(tgt_patch, tgt_objs, "TGT")

        matched, unmatched_ref_idx, unmatched_tgt_idx = self._match(ref_objs, tgt_objs, img_diag)

        removed_candidates = [ref_objs[i] for i in unmatched_ref_idx]
        confirmed_rm_local, rescued_rm_local = self._double_check(removed_candidates, tgt_patch, "REMOVED")
        confirmed_ref = [unmatched_ref_idx[i] for i in confirmed_rm_local]
        rescued_ref = [unmatched_ref_idx[i] for i in rescued_rm_local]

        added_candidates = [tgt_objs[i] for i in unmatched_tgt_idx]
        confirmed_add_local, rescued_add_local = self._double_check(added_candidates, ref_patch, "ADDED")
        confirmed_tgt = [unmatched_tgt_idx[i] for i in confirmed_add_local]
        rescued_tgt = [unmatched_tgt_idx[i] for i in rescued_add_local]

        removed_objs = [ref_objs[i] for i in confirmed_ref]
        added_objs = [tgt_objs[i] for i in confirmed_tgt]

        logger.info(
            f"After double-check → "
            f"removed={len(removed_objs)} (rescued={len(rescued_ref)}), "
            f"added={len(added_objs)} (rescued={len(rescued_tgt)})"
        )

        result = {
            "roi": list(roi),
            "ref_image": ref_path,
            "target_image": tgt_path,
            "ref_object_count": len(ref_objs),
            "target_object_count": len(tgt_objs),
            "matched_count": len(matched),
            "rescued_count": len(rescued_ref) + len(rescued_tgt),
            "added_count": len(added_objs),
            "removed_count": len(removed_objs),
            "changes": {
                "added": [{"class": o.class_name, "confidence": float(o.confidence), "box": o.box.tolist()}
                          for o in added_objs],
                "removed": [{"class": o.class_name, "confidence": float(o.confidence), "box": o.box.tolist()}
                            for o in removed_objs],
                "matched": [{"ref_class": ref_objs[r].class_name,
                             "tgt_class": tgt_objs[t].class_name,
                             "score": float(s)} for t, r, s in matched],
            },
        }

        if self.cfg.save_visualization:
            vis_path = str(roi_output_dir / "diff.jpg")
            self._visualize(
                ref_patch, tgt_patch,
                ref_objs, tgt_objs,
                matched,
                confirmed_ref, confirmed_tgt,
                rescued_ref, rescued_tgt,
                vis_path,
            )

        return result

    # ── Public API ────────────────────────────────────────────────────────────
    def compare(
        self,
        ref_path: str,
        tgt_path: str,
        rois: Optional[list[tuple[int, int, int, int]]] = None,
    ) -> dict:
        ref_img = cv2.imread(ref_path)
        tgt_img = cv2.imread(tgt_path)
        if ref_img is None or tgt_img is None:
            raise FileNotFoundError(f"Image not found: ref={ref_path}, tgt={tgt_path}")

        if rois is None:
            rois = load_rois_from_file(self.cfg.roi_file)
        if rois is None:
            logger.info("ROI was not loaded from file — opening the GUI.")
            rois = ROISelector(ref_img).run()
        if not rois:
            raise ValueError("No ROI has been defined.")

        reference_cache = self._prepare_reference_cache(ref_img_full=ref_img, rois=rois, ref_path=ref_path)

        t_start = time.perf_counter()
        run_dir = next_run_dir(self.cfg.base_output_dir)
        all_results = []

        for roi_idx, roi in enumerate(rois):
            logger.info(f"═══ ROI {roi_idx + 1}/{len(rois)}: {roi} ═══")
            roi_dir = run_dir / f"roi_{roi_idx + 1}"
            roi_dir.mkdir(parents=True, exist_ok=True)

            result = self._compare_roi(
                ref_img,
                tgt_img,
                roi,
                reference_cache[roi_idx],
                roi_dir,
                ref_path,
                tgt_path,
            )
            all_results.append(result)

        total_added = sum(r["added_count"] for r in all_results)
        total_removed = sum(r["removed_count"] for r in all_results)
        total_matched = sum(r["matched_count"] for r in all_results)
        total_rescued = sum(r["rescued_count"] for r in all_results)
        elapsed_sec = time.perf_counter() - t_start

        summary = {
            "ref_image": ref_path,
            "target_image": tgt_path,
            "total_rois": len(rois),
            "total_matched": total_matched,
            "total_rescued": total_rescued,
            "total_added": total_added,
            "total_removed": total_removed,
            "has_changes": total_added > 0 or total_removed > 0,
            "elapsed_sec": round(elapsed_sec, 2),
            "timing_note": "Measured after reference cache preparation/reuse.",
            "roi_results": all_results,
        }

        summary_path = run_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=json_serializer)
        logger.info(f"Summary → {summary_path}")
        logger.info(
            f"DONE | added={total_added} removed={total_removed} "
            f"matched={total_matched} rescued={total_rescued} "
            f"time={summary['elapsed_sec']}s "
            f"(reference cache excluded)"
        )
        return summary

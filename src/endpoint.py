# src/endpoint.py
import json
import logging
import tempfile
import os
from typing import List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile

from .config import DetectorConfig
from .processor import ChangeDetector

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Change Detection API",
    description="API for comparing images and detecting changes/matching objects",
    version="1.0.0",
)

router = APIRouter()

default_cfg = DetectorConfig()
detector: Optional[ChangeDetector] = None


def get_detector() -> ChangeDetector:
    global detector
    if detector is None:
        logger.info("Initializing ChangeDetector instance with DetectorConfig...")
        detector = ChangeDetector(config=default_cfg)
    return detector


@app.on_event("startup")
async def startup_event():
    try:
        get_detector()
        logger.info("ChangeDetector loaded successfully on startup.")
    except Exception as e:
        logger.error(f"Failed to pre-load detector on startup: {e}")


@router.post("/compare")
async def compare_images(
    file: UploadFile = File(..., description="Target image file"),
    reference_image_path: Optional[str] = Form(None),
    rois_json: Optional[str] = Form(None),
):
    det = get_detector()

    try:
        image_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {e}")

    parsed_rois: Optional[List[List[int]]] = None
    if rois_json:
        try:
            parsed_rois = json.loads(rois_json)
            if not isinstance(parsed_rois, list):
                raise ValueError("ROIs must be a list.")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid rois_json: {e}")

    final_ref_path = reference_image_path or default_cfg.reference_image_path

    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or ".jpg")[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        result = det.compare(
            ref_path=final_ref_path,
            tgt_path=tmp_path,
            rois=parsed_rois,
        )
        return {"status": "success", "filename": file.filename, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during change detection processing")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


app.include_router(router)

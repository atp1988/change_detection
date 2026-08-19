import logging
import argparse

import torch
import uvicorn

from .config import DetectorConfig
from .processor import ChangeDetector
from .utils import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("change_detection.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("ChangeDetector")


def run_cli(args):
    cfg = DetectorConfig(
        yolo_model=args.model,
        roi_file=args.roi_file,
        similarity_threshold=args.sim_threshold,
        double_check_threshold=args.double_check_threshold,
        base_output_dir=args.output_dir,
        save_visualization=not args.no_vis,
        reference_image_path=args.ref,  # CLI overrides config reference path
    )

    # Ensure compares/runs exists in project root (as requested)
    ensure_dir(cfg.base_output_dir)

    detector = ChangeDetector(cfg)

    print("YOLO device:", next(detector.models.yolo.model.parameters()).device)
    print("DINO device:", next(detector.models.dino.parameters()).device)
    print("GPU memory used (MB):", torch.cuda.memory_allocated() / 1024**2)

    result = detector.compare(cfg.reference_image_path, args.tgt)

    print("\n" + "═" * 50)
    print(f"  Detected changes: {'Yes ✓' if result['has_changes'] else 'No ✗'}")
    print(f"  Added     : {result['total_added']}")
    print(f"  Removed   : {result['total_removed']}")
    print(f"  Matched   : {result['total_matched']}")
    print(f"  Rescued   : {result['total_rescued']}")
    print(f"  Time      : {result['elapsed_sec']}s")
    print("═" * 50)


def run_api(args):
    # Runs FastAPI app
    uvicorn.run(
        "src.endpoint:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Change Detection System")
    sub = parser.add_subparsers(dest="mode", required=True)

    # CLI mode (kept similar to your original script)
    cli = sub.add_parser("cli", help="Run once from CLI (like original)")
    cli.add_argument("ref", help="Reference image path")
    cli.add_argument("tgt", help="Target image path")
    cli.add_argument("--model", default="models/yolov8s-worldv2.pt",
                     help="YOLO model path (default: models/yolov8s-worldv2.pt)")
    cli.add_argument("--roi-file", default="roi.json",
                     help="ROI file path (default: roi.json)")
    cli.add_argument("--sim-threshold", type=float, default=0.55,
                     help="Similarity threshold for matching")
    cli.add_argument("--double-check-threshold", type=float, default=0.60,
                     help="Double-check threshold for false alarms")
    cli.add_argument("--output-dir", default="compares/runs",
                     help="Output directory")
    cli.add_argument("--no-vis", action="store_true",
                     help="Disable saving the diff image")

    # API mode
    api = sub.add_parser("api", help="Run FastAPI server")
    api.add_argument("--host", default="0.0.0.0")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true")

    args = parser.parse_args()

    if args.mode == "cli":
        run_cli(args)
    elif args.mode == "api":
        run_api(args)

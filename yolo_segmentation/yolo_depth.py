"""Compare YOLO26 and Depth Anything V2 depth estimation on camera frames."""

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils.torch_utils import select_device

from yolo_seg import write_image


MODEL_DIR = Path(__file__).resolve().parent / "models"
DA_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
WINDOW_NAME = "Depth Comparison - Q/ESC: quit, S: save"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--backend", choices=("both", "yolo", "depth-anything"), default="both")
    parser.add_argument("--model", default=str(MODEL_DIR / "yolo26n-depth.pt"),
                        help="YOLO depth weights (default: models\\yolo26n-depth.pt)")
    parser.add_argument("--da-model", default=DA_MODEL,
                        help="Depth Anything V2 Hugging Face model ID or local directory")
    parser.add_argument("--imgsz", type=int, default=768,
                        help="YOLO inference size (default: 768; DA uses its processor config)")
    parser.add_argument("--device", default="0",
                        help="Inference device (default: 0, CUDA GPU); use cpu for CPU")
    parser.add_argument("--save-dir", type=Path, default=Path("captures") / "depth")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    return args


class DepthAnythingEstimator:
    def __init__(self, model_id: str, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(
            model_id, cache_dir=str(MODEL_DIR / "huggingface"), use_fast=False
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, cache_dir=str(MODEL_DIR / "huggingface")
        ).to(device).eval()

    def predict(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth
            depth = torch.nn.functional.interpolate(
                predicted.unsqueeze(1), size=frame.shape[:2],
                mode="bicubic", align_corners=False,
            )[0, 0]
        return depth.float().cpu().numpy()


def colorize_depth(depth: np.ndarray, shape: tuple[int, int], inverse: bool) -> np.ndarray:
    """Normalize each map independently, with warm colors indicating nearer surfaces."""
    if depth.shape != shape:
        raise RuntimeError(f"Expected original-resolution depth {shape}, got {depth.shape}")
    if not np.isfinite(depth).all():
        raise RuntimeError("Depth map contains non-finite values")
    minimum, maximum = float(depth.min()), float(depth.max())
    normalized = np.zeros(shape, dtype=np.uint8)
    if maximum > minimum:
        values = (depth - minimum) / (maximum - minimum)
        if not inverse:
            values = 1.0 - values
        normalized = (values * 255).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)


def make_preview(frame: np.ndarray, colors: dict[str, np.ndarray]) -> np.ndarray:
    panels = []
    for label, image in {"Camera": frame, **colors}.items():
        panel = image.copy()
        cv2.putText(panel, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def save_depths(
    frame: np.ndarray, depths: dict[str, np.ndarray],
    colors: dict[str, np.ndarray], save_dir: Path,
) -> Path:
    destination = save_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    write_image(destination / "frame.png", frame)
    write_image(destination / "comparison.png", make_preview(frame, colors))
    for name, depth in depths.items():
        np.save(destination / f"{name}_depth.npy", depth, allow_pickle=False)
        write_image(destination / f"{name}_color.png", colors[name])
    print(f"Saved depth maps to {destination} (raw predictions, not calibrated meters)")
    return destination


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device, verbose=False)
    print(f"Inference device: {device}")
    yolo = YOLO(args.model) if args.backend in ("both", "yolo") else None
    if yolo is not None and yolo.task != "depth":
        raise ValueError(f"Use depth weights, not a {yolo.task!r} model")
    da = DepthAnythingEstimator(args.da_model, device) if args.backend in (
        "both", "depth-anything"
    ) else None

    camera = cv2.VideoCapture(args.camera)
    durations: deque[float] = deque(maxlen=30)
    fps = None
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera}. Check the camera index, "
                "Windows camera permissions, and other apps using the camera."
            )
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1440, 480)
        print("Q / ESC: quit | S: save frame, raw depth NPY, and color maps")
        print("Warm = near; maps normalized independently per frame, not calibrated meters.")
        while True:
            started = perf_counter()
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read a frame from camera {args.camera}")
            depths = {}
            colors = {}
            if yolo is not None:
                result = next(iter(yolo.predict(
                    source=frame, device=str(device), imgsz=args.imgsz, verbose=False,
                )))
                if not isinstance(result, Results) or result.depth is None:
                    raise RuntimeError("Expected depth results; use a YOLO depth model")
                depths["yolo"] = np.asarray(result.depth.cpu().numpy().data, dtype=np.float32)
                colors["yolo"] = colorize_depth(depths["yolo"], frame.shape[:2], inverse=False)
            if da is not None:
                depths["depth_anything"] = da.predict(frame)
                colors["depth_anything"] = colorize_depth(
                    depths["depth_anything"], frame.shape[:2], inverse=True
                )
            preview = make_preview(frame, colors)
            resolution = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
            fps_label = "FPS: warming up" if fps is None else f"FPS: {fps:.1f}"
            cv2.putText(preview, resolution, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, fps_label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(preview, f"Device: {device}", (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            durations.append(perf_counter() - started)
            fps = len(durations) / sum(durations)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save_depths(frame, depths, colors, args.save_dir)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

"""Run YOLO oriented bounding box detection on OpenCV camera frames."""

import argparse
from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from yolo_seg import write_image


WINDOW_NAME = "YOLO OBB - Q/ESC: quit, S: save"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "yolo26n-obb.pt"),
        help="OBB weights (default: models\\yolo26n-obb.pt beside script)",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    parser.add_argument("--imgsz", type=int, default=1024, help="Inference size (default: 1024)")
    parser.add_argument("--save-dir", type=Path, default=Path("captures") / "obb")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    if not 0.0 < args.conf <= 1.0:
        parser.error("--conf must be greater than 0 and at most 1")
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    return args


def save_detections(frame: np.ndarray, result: Results, save_dir: Path) -> Path:
    """Save images and original-image pixel coordinates, with rotation in radians."""
    if result.obb is None:
        raise RuntimeError("Expected OBB results; use an OBB model")
    destination = save_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    write_image(destination / "frame.png", frame)
    write_image(destination / "obb.png", result.plot())
    boxes = result.obb.cpu()
    detections = []
    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        detections.append({
            "class_id": class_id,
            "class_name": result.names[class_id],
            "confidence": float(boxes.conf[index].item()),
            "xywhr": boxes.xywhr[index].tolist(),
            "corners_xy": boxes.xyxyxyxy[index].tolist(),
        })
    payload = {
        "image_width": frame.shape[1],
        "image_height": frame.shape[0],
        "coordinate_units": "pixels",
        "rotation_units": "radians",
        "detections": detections,
    }
    with (destination / "detections.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Saved {len(detections)} oriented boxes to {destination}")
    return destination


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    if model.task != "obb":
        raise ValueError(f"Use OBB weights, not a {model.task!r} model")
    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera}. Check the camera index, "
                "Windows camera permissions, and other apps using the camera."
            )
        print("Q / ESC: quit | S: save current frame, oriented boxes, and JSON")
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read a frame from camera {args.camera}")
            result = model.predict(
                source=frame, conf=args.conf, device=args.device,
                imgsz=args.imgsz, verbose=False,
            )[0]
            cv2.imshow(WINDOW_NAME, result.plot())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save_detections(frame, result, args.save_dir)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

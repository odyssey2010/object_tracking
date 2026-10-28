"""Run YOLO instance segmentation on OpenCV camera frames."""

import argparse
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


WINDOW_NAME = "YOLO Segmentation - Q/ESC: quit, S: save"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--model",
        default=str(Path(__file__).resolve().parent / "models" / "yolo26n-seg.pt"),
        help="Segmentation weights (default: models\\yolo26n-seg.pt beside script)",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device, e.g. cpu or 0")
    parser.add_argument("--save-dir", type=Path, default=Path("captures"))
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    if not 0.0 < args.conf <= 1.0:
        parser.error("--conf must be greater than 0 and at most 1")
    return args


def write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def save_instances(frame: np.ndarray, result: Results, save_dir: Path) -> Path:
    """Save original-resolution masks and objects from the unannotated frame."""
    destination = save_dir / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination.mkdir(parents=True, exist_ok=False)
    write_image(destination / "frame.png", frame)
    write_image(destination / "segmentation.png", result.plot())
    if result.masks is None:
        print(f"No objects detected; saved frame to {destination}")
        return destination
    if result.boxes is None or len(result.boxes) != len(result.masks):
        raise RuntimeError("Segmentation masks and detection boxes do not match")

    height, width = frame.shape[:2]
    masks = result.masks.data.cpu().numpy()
    for index, mask in enumerate(masks):
        if mask.shape != (height, width):
            raise RuntimeError("Expected original-resolution masks; use retina_masks=True")
        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        class_id = int(result.boxes.cls[index].item())
        prefix = f"object_{index:03d}_class_{class_id}"
        isolated = cv2.bitwise_and(frame, frame, mask=binary_mask)
        write_image(destination / f"{prefix}_mask.png", binary_mask)
        write_image(destination / f"{prefix}_isolated.png", isolated)
    print(f"Saved {len(masks)} objects to {destination}")
    return destination


def run(args: argparse.Namespace) -> None:
    model = YOLO(args.model)
    if model.task != "segment":
        raise ValueError(f"Use segmentation weights, not a {model.task!r} model")

    camera = cv2.VideoCapture(args.camera)
    try:
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera}. Check the camera index, "
                "Windows camera permissions, and other apps using the camera."
            )
        print("Q / ESC: quit | S: save current frame, masks, and isolated objects")
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read a frame from camera {args.camera}")
            result = model.predict(
                source=frame,
                conf=args.conf,
                device=args.device,
                retina_masks=True,
                verbose=False,
            )[0]
            cv2.imshow(WINDOW_NAME, result.plot())
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("s"), ord("S")):
                save_instances(frame, result, args.save_dir)
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

"""Run Depth Anything V2 on camera frames with live processing FPS."""

import argparse
from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
from ultralytics.utils.torch_utils import select_device

from yolo_depth import (
    DA_MODEL,
    DepthAnythingEstimator,
    colorize_depth,
    make_preview,
    save_depths,
)


WINDOW_NAME = "Depth Anything V2 - Q/ESC: quit, S: save"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", default=DA_MODEL,
                        help="Depth Anything V2 Hugging Face model ID or local directory")
    parser.add_argument("--device", default="0",
                        help="Inference device (default: 0, CUDA GPU); use cpu for CPU")
    parser.add_argument("--save-dir", type=Path, default=Path("captures") / "depth_anything")
    args = parser.parse_args()
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    return args


def run(args: argparse.Namespace) -> None:
    device = select_device(args.device, verbose=False)
    print(f"Inference device: {device}")
    estimator = DepthAnythingEstimator(args.model, device)
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
        cv2.resizeWindow(WINDOW_NAME, 1280, 480)
        print("Q / ESC: quit | S: save frame, raw depth NPY, and color map")
        print("Warm = near; relative depth, not calibrated meters.")
        print("FPS: rolling 30-frame processing rate, including camera and display.")
        last_report = perf_counter()
        while True:
            started = perf_counter()
            success, frame = camera.read()
            if not success or frame is None:
                raise RuntimeError(f"Could not read a frame from camera {args.camera}")
            depth = estimator.predict(frame)
            colors = {
                "depth_anything": colorize_depth(depth, frame.shape[:2], inverse=True)
            }
            preview = make_preview(frame, colors)
            label = "FPS: warming up" if fps is None else f"FPS: {fps:.1f}"
            cv2.putText(preview, label, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("s"), ord("S")):
                save_depths(frame, {"depth_anything": depth}, colors, args.save_dir)
            closed = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
            finished = perf_counter()
            durations.append(finished - started)
            fps = len(durations) / sum(durations)
            if finished - last_report >= 1.0:
                print(f"FPS: {fps:.1f}")
                last_report = finished
            if key in (ord("q"), ord("Q"), 27) or closed:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run(parse_args())

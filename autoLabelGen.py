import os
import cv2
import time
import argparse
import logging
import threading
import queue as queue_module
from pathlib import Path
from ultralytics import YOLO

# Optional ROS2 support
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image as RosImage
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def to_yolo_label(frame, box):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


# === Input sources ===

def iter_camera(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {source}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def iter_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame
    finally:
        cap.release()


def iter_folder(folder):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)
    if not paths:
        raise RuntimeError(f"No images found in folder: {folder}")
    for p in paths:
        frame = cv2.imread(str(p))
        if frame is not None:
            yield frame


def iter_ros_topic(topic):
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS2 / cv_bridge not available. Install ros-humble-cv-bridge.")

    frame_queue = queue_module.Queue(maxsize=5)
    bridge = CvBridge()

    class FrameNode(Node):
        def __init__(self):
            super().__init__('auto_label_subscriber')
            self.create_subscription(RosImage, topic, self._cb, 10)

        def _cb(self, msg):
            try:
                frame = bridge.imgmsg_to_cv2(msg, 'bgr8')
                if not frame_queue.full():
                    frame_queue.put_nowait(frame)
            except Exception as e:
                self.get_logger().error(f"cv_bridge error: {e}")

    rclpy.init()
    node = FrameNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(f"Subscribed to ROS topic: {topic}")
    try:
        while rclpy.ok():
            try:
                yield frame_queue.get(timeout=1.0)
            except queue_module.Empty:
                continue
    finally:
        node.destroy_node()
        rclpy.shutdown()


def get_frame_iter(source_type, source):
    if source_type == 'camera':
        return iter_camera(source)
    elif source_type == 'video':
        return iter_video(source)
    elif source_type == 'folder':
        return iter_folder(source)
    elif source_type == 'ros':
        return iter_ros_topic(source)
    raise ValueError(f"Unknown source type: {source_type}")


def next_img_index(folder):
    """Return the next available img index based on existing files."""
    indices = []
    for f in Path(folder).glob('img*.jpg'):
        try:
            indices.append(int(f.stem[3:]))
        except ValueError:
            pass
    return max(indices) + 1 if indices else 0


# === Main loop ===

def run(model_path, source_type, source, out_images, out_labels,
        conf=0.5, blur_thresh=25.0, show=False):

    logging.getLogger('ultralytics').setLevel(logging.ERROR)
    model = YOLO(model_path)

    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    img_idx = next_img_index(out_images)
    saved = 0
    print(f"Model: {model_path}")
    print(f"Source: {source_type} — {source}")
    print(f"Output: {out_images} / {out_labels}")
    print(f"Starting at img{img_idx}. Press Q to stop.\n")

    try:
        for frame in get_frame_iter(source_type, source):
            results = model(frame, conf=conf, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                continue

            label_lines = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cls = int(box.cls[0])

                if blur_thresh > 0:
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    if crop.size == 0:
                        continue
                    score = blur_score(crop)
                    if score < blur_thresh:
                        print(f"img{img_idx}: blur {score:.1f} below threshold, skipping crop")
                        continue

                cx, cy, bw, bh = to_yolo_label(frame, (x1, y1, x2, y2))
                label_lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if not label_lines:
                continue

            img_path = os.path.join(out_images, f"img{img_idx}.jpg")
            lbl_path = os.path.join(out_labels, f"img{img_idx}.txt")
            cv2.imwrite(img_path, frame)
            with open(lbl_path, 'w') as f:
                f.write('\n'.join(label_lines) + '\n')

            print(f"Saved img{img_idx}.jpg — {len(label_lines)} detection(s)")
            saved += 1
            img_idx += 1

            if show:
                display = frame.copy()
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, str(cls), (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Auto Label", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if show:
            cv2.destroyAllWindows()
        print(f"\nDone. Saved {saved} labeled image(s) to '{out_images}'.")


if __name__ == '__main__':
    def_model_path   = "models/best_alex.pt"
    def_source       = "/zed/zed_node/rgb/color/rect/image"
    def_source_type  = "ros"
    def_source_type  = "folder"
    def_source       = "buoy_images"
    def_out_images   = "trainImagesZed/images"
    def_out_labels   = "trainImagesZed/labels"
    def_conf         = 0.5
    def_blur         = 25.0

    parser = argparse.ArgumentParser(description="Generic YOLO auto-labeling tool")
    parser.add_argument('-m', '--model', default=def_model_path,
                        help=f"Path to YOLO .pt model (default: {def_model_path})")
    parser.add_argument('-s', '--source', default=def_source,
                        help=f"Camera index, video path, image folder, or ROS topic (default: {def_source})")
    parser.add_argument('-t', '--source-type', dest='source_type',
                        choices=['camera', 'video', 'folder', 'ros'], default=def_source_type,
                        help=f"Input source type (default: {def_source_type})")
    parser.add_argument('--out-images', default=def_out_images,
                        help=f"Output folder for images (default: {def_out_images})")
    parser.add_argument('--out-labels', default=def_out_labels,
                        help=f"Output folder for labels (default: {def_out_labels})")
    parser.add_argument('-c', '--conf', type=float, default=def_conf,
                        help=f"Detection confidence threshold (default: {def_conf})")
    parser.add_argument('--blur', type=float, default=def_blur,
                        help=f"Blur threshold (Laplacian variance); 0 to disable (default: {def_blur})")
    parser.add_argument('-v', '--show', action='store_true',
                        help="Show detections in a live window")
    args = parser.parse_args()

    run(args.model, args.source_type, args.source,
        args.out_images, args.out_labels,
        args.conf, args.blur, args.show)

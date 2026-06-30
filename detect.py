

import argparse
import time
from pathlib import Path

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

# ─── Config ──────────────────────────────────────────────────────────────────
VEHICLE_CLASSES   = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONF_THRESHOLD    = 0.4          # min YOLO confidence
IOU_THRESHOLD     = 0.4          # IOU to match same car across frames
LINE_RATIO        = 0.55         # counting line position (fraction of height)
SPEED_SCALE       = 0.05         # pixels-per-frame → arbitrary speed units
PLATE_EXPAND      = 0.15         # expand plate crop for better OCR
FONT              = cv2.FONT_HERSHEY_SIMPLEX

# ─── Colours ─────────────────────────────────────────────────────────────────
C_BOX   = (0,   200,  0)
C_PLATE = (0,   165, 255)
C_LINE  = (0,    0,  220)
C_TEXT  = (255, 255, 255)
C_SHADOW= (0,    0,   0)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def iou(boxA, boxB):
    """Intersection over Union of two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    aA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    aB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    return inter / (aA + aB - inter)


def draw_text(img, text, x, y, scale=0.6, thickness=2, color=C_TEXT):
    cv2.putText(img, text, (x+1, y+1), FONT, scale, C_SHADOW, thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y  ), FONT, scale, color,    thickness,   cv2.LINE_AA)


def crop_expand(img, x1, y1, x2, y2, expand=PLATE_EXPAND):
    h, w = img.shape[:2]
    dx = int((x2 - x1) * expand); dy = int((y2 - y1) * expand)
    cx1 = max(0, x1 - dx); cy1 = max(0, y1 - dy)
    cx2 = min(w, x2 + dx); cy2 = min(h, y2 + dy)
    return img[cy1:cy2, cx1:cx2]


# ─── Core Processor ──────────────────────────────────────────────────────────

class Detector:
    def __init__(self):
        print("[*] Loading YOLOv8n …")
        self.yolo  = YOLO("yolov8n.pt")          # auto-downloaded on first run
        print("[*] Loading EasyOCR …")
        self.ocr   = easyocr.Reader(["en"], gpu=False)
        self.tracks: dict[int, dict] = {}        # id → {box, plate, speed, counted}
        self.next_id = 0
        self.count_in  = 0
        self.count_out = 0
        self.line_y: int | None = None

    # ── per-frame OCR on a small crop ──
    def read_plate(self, crop) -> str:
        if crop.size == 0:
            return ""
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        results = self.ocr.readtext(gray, detail=1, paragraph=False)
        if not results:
            return ""
        # pick highest confidence result
        best = max(results, key=lambda r: r[2])
        return best[1].upper().strip() if best[2] > 0.25 else ""

    # ── match detection to existing track via IOU ──
    def match_track(self, box):
        best_id, best_iou = None, IOU_THRESHOLD
        for tid, t in self.tracks.items():
            score = iou(box, t["box"])
            if score > best_iou:
                best_iou, best_id = score, tid
        return best_id

    # ── update counting line ──
    def update_line(self, frame_h):
        self.line_y = int(frame_h * LINE_RATIO)

    # ── process one frame / image ──
    def process_frame(self, frame, do_ocr=True):
        h, w = frame.shape[:2]
        self.update_line(h)

        results = self.yolo(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        current_ids = set()

        for box_data in results.boxes:
            cls   = int(box_data.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box_data.xyxy[0])
            box = [x1, y1, x2, y2]
            cy  = (y1 + y2) // 2
            cx  = (x1 + x2) // 2

            # ── track matching ──
            tid = self.match_track(box)
            if tid is None:
                tid = self.next_id; self.next_id += 1
                self.tracks[tid] = {
                    "box": box, "plate": "", "prev_cy": cy,
                    "speed": 0.0, "counted": False,
                    "label": VEHICLE_CLASSES[cls]
                }

            track = self.tracks[tid]
            # speed (pixels/frame scaled)
            track["speed"] = abs(cy - track["prev_cy"]) * SPEED_SCALE * 100
            track["prev_cy"] = cy
            track["box"] = box

            # ── in/out counting ──
            if not track["counted"] and self.line_y:
                if abs(cy - self.line_y) < 20:
                    track["counted"] = True
                    if cx < w // 2:
                        self.count_out += 1
                    else:
                        self.count_in  += 1

            # ── OCR on plate region (lower 35% of vehicle box) ──
            if do_ocr:
                ph = y2 - y1
                py1 = y1 + int(ph * 0.60)
                plate_crop = crop_expand(frame, x1, py1, x2, y2)
                plate_text = self.read_plate(plate_crop)
                if plate_text:
                    track["plate"] = plate_text

            current_ids.add(tid)

            # ── draw vehicle box ──
            cv2.rectangle(frame, (x1, y1), (x2, y2), C_BOX, 2)
            label = f"{track['label']}  {track['speed']:.1f} u/s"
            draw_text(frame, label, x1, y1 - 8)

            # ── draw plate text ──
            if track["plate"]:
                plate_w = (x2 - x1)
                px1b = x1; px2b = x1 + plate_w
                py1b = y2 - int((y2-y1)*0.28)
                cv2.rectangle(frame, (px1b, py1b), (px2b, y2), C_PLATE, 2)
                draw_text(frame, track["plate"], px1b + 4, py1b - 6,
                          scale=0.55, color=(0, 220, 255))

        # ── remove stale tracks ──
        self.tracks = {k: v for k, v in self.tracks.items() if k in current_ids}

        # ── draw counting line ──
        if self.line_y:
            cv2.line(frame, (0, self.line_y), (w, self.line_y), C_LINE, 2)

        # ── overlay counters ──
        cv2.rectangle(frame, (8, 8), (200, 70), (0, 0, 0), -1)
        draw_text(frame, f"IN : {self.count_in}",  14, 32, scale=0.7, color=(100,255,100))
        draw_text(frame, f"OUT: {self.count_out}", 14, 60, scale=0.7, color=(100,200,255))

        return frame


# ─── Image mode ──────────────────────────────────────────────────────────────

def process_image(input_path: str, output_path: str | None):
    det   = Detector()
    frame = cv2.imread(input_path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {input_path}")

    print(f"[*] Processing image {input_path} …")
    out = det.process_frame(frame, do_ocr=True)

    if output_path:
        cv2.imwrite(output_path, out)
        print(f"[✓] Saved to {output_path}")
    else:
        cv2.imshow("Result — press any key to close", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"[i] IN={det.count_in}  OUT={det.count_out}")


# ─── Video mode ──────────────────────────────────────────────────────────────

def process_video(input_path: str, output_path: str | None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    det = Detector()
    frame_n = 0
    t0 = time.time()

    print(f"[*] Processing video {input_path}  ({W}×{H} @ {fps:.1f}fps, {total} frames) …")
    print("[i] Press Q to quit early.\n")

    # Run OCR every N frames to keep things fast
    OCR_EVERY = max(1, int(fps // 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        do_ocr = (frame_n % OCR_EVERY == 0)
        out = det.process_frame(frame, do_ocr=do_ocr)

        # progress
        if frame_n % 30 == 0:
            elapsed = time.time() - t0
            pct = frame_n / total * 100 if total else 0
            print(f"  frame {frame_n}/{total} ({pct:.0f}%)  "
                  f"IN={det.count_in} OUT={det.count_out}  "
                  f"elapsed {elapsed:.1f}s")

        if writer:
            writer.write(out)
        else:
            cv2.imshow("Detection — Q to quit", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
        print(f"[✓] Saved to {output_path}")
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print(f"\n[✓] Done in {elapsed:.1f}s")
    print(f"[i] Final counts — IN: {det.count_in}  OUT: {det.count_out}")


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vehicle detection, plate OCR, speed & counting")
    parser.add_argument("--input",  required=True, help="Path to image or video file")
    parser.add_argument("--output", default=None,  help="Output file path (optional)")
    args = parser.parse_args()

    ext = Path(args.input).suffix.lower()
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        process_image(args.input, args.output)
    elif ext in {".mp4", ".avi", ".mov", ".mkv", ".wmv"}:
        process_video(args.input, args.output)
    else:
        print(f"[!] Unsupported file type: {ext}")


if __name__ == "__main__":
    main()

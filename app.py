"""
Vehicle Detection Web App — Flask Backend (fast edition)
=========================================================
Install:
    pip install flask ultralytics easyocr opencv-python numpy

Run:
    python app.py
    Then open http://localhost:5000
"""

import os, time, uuid, threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory

# ─── Config ──────────────────────────────────────────────────────────────────
UPLOAD_FOLDER   = "uploads"
OUTPUT_FOLDER   = "outputs"
ALLOWED_IMG     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ALLOWED_VID     = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
MAX_SIZE_MB     = 200

VEHICLE_CLASSES = {2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
CONF_THRESHOLD  = 0.4
IOU_THRESHOLD   = 0.4
LINE_RATIO      = 0.55
SPEED_SCALE     = 0.05
PLATE_EXPAND    = 0.15
YOLO_INPUT_W    = 640        # resize width fed to YOLO (speed vs accuracy)
YOLO_EVERY      = 3          # run YOLO every N frames
OCR_EVERY       = 10         # run OCR every N frames (when enabled)
FONT            = cv2.FONT_HERSHEY_SIMPLEX

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = MAX_SIZE_MB * 1024 * 1024

_yolo = _ocr = None
_lock = threading.Lock()

def get_models(need_ocr=False):
    global _yolo, _ocr
    with _lock:
        if _yolo is None:
            from ultralytics import YOLO
            _yolo = YOLO("yolov8n.pt")
        if need_ocr and _ocr is None:
            import easyocr
            _ocr = easyocr.Reader(["en"], gpu=False)
    return _yolo, (_ocr if need_ocr else None)

jobs: dict[str, dict] = {}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def iou(a, b):
    xA,yA = max(a[0],b[0]),max(a[1],b[1])
    xB,yB = min(a[2],b[2]),min(a[3],b[3])
    inter = max(0,xB-xA)*max(0,yB-yA)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def draw_text(img, text, x, y, scale=0.55, thick=2, color=(255,255,255)):
    cv2.putText(img,text,(x+1,y+1),FONT,scale,(0,0,0),thick+1,cv2.LINE_AA)
    cv2.putText(img,text,(x,  y  ),FONT,scale,color,  thick,  cv2.LINE_AA)

def crop_expand(img, x1, y1, x2, y2, ex=PLATE_EXPAND):
    h,w = img.shape[:2]
    dx=int((x2-x1)*ex); dy=int((y2-y1)*ex)
    return img[max(0,y1-dy):min(h,y2+dy), max(0,x1-dx):min(w,x2+dx)]

def read_plate(ocr_model, crop):
    if crop.size==0 or ocr_model is None: return ""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    results = ocr_model.readtext(gray, detail=1, paragraph=False)
    if not results: return ""
    best = max(results, key=lambda r: r[2])
    return best[1].upper().strip() if best[2]>0.25 else ""

# ─── Core processor (stateful, call per frame) ────────────────────────────────

class VideoProcessor:
    def __init__(self, yolo, ocr, orig_w, orig_h, enable_ocr):
        self.yolo = yolo
        self.ocr  = ocr
        self.enable_ocr = enable_ocr
        self.orig_w = orig_w
        self.orig_h = orig_h
        # scale factor: we feed a smaller frame to YOLO
        self.scale = min(1.0, YOLO_INPUT_W / orig_w)
        self.inf_w = int(orig_w * self.scale)
        self.inf_h = int(orig_h * self.scale)
        self.tracks: dict = {}
        self.counts = {"in":0,"out":0}
        self.line_y = int(orig_h * LINE_RATIO)
        self.last_boxes: list = []   # reused between YOLO frames
        self.all_plates: set = set()
        self.fn = 0

    def process(self, frame):
        self.fn += 1
        run_yolo = (self.fn % YOLO_EVERY == 1)
        run_ocr  = self.enable_ocr and (self.fn % OCR_EVERY == 1)

        if run_yolo:
            small = cv2.resize(frame, (self.inf_w, self.inf_h))
            results = self.yolo(small, conf=CONF_THRESHOLD, verbose=False)[0]
            s = self.scale
            self.last_boxes = []
            for bd in results.boxes:
                cls = int(bd.cls[0])
                if cls not in VEHICLE_CLASSES: continue
                x1,y1,x2,y2 = (int(v/s) for v in bd.xyxy[0])
                self.last_boxes.append((cls,x1,y1,x2,y2))

        seen = set()
        for (cls,x1,y1,x2,y2) in self.last_boxes:
            box = [x1,y1,x2,y2]
            cy  = (y1+y2)//2; cx = (x1+x2)//2

            # track matching
            tid, best = None, IOU_THRESHOLD
            for k,t in self.tracks.items():
                s = iou(box, t["box"])
                if s>best: best,tid = s,k
            if tid is None:
                tid = len(self.tracks)+self.fn  # unique id
                self.tracks[tid] = {"box":box,"plate":"","prev_cy":cy,
                                    "speed":0.0,"counted":False,
                                    "label":VEHICLE_CLASSES[cls]}
            t = self.tracks[tid]
            t["speed"]   = abs(cy-t["prev_cy"])*SPEED_SCALE*100
            t["prev_cy"] = cy
            t["box"]     = box
            seen.add(tid)

            # counting
            if not t["counted"] and abs(cy-self.line_y)<22:
                t["counted"] = True
                if cx < self.orig_w//2: self.counts["out"] += 1
                else:                   self.counts["in"]  += 1

            # OCR
            if run_ocr:
                ph   = y2-y1
                crop = crop_expand(frame, x1, y1+int(ph*0.60), x2, y2)
                txt  = read_plate(self.ocr, crop)
                if txt:
                    t["plate"] = txt
                    self.all_plates.add(txt)

            # draw vehicle box
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,200,0),2)
            lbl = f"{t['label']}  {t['speed']:.0f}u/s"
            draw_text(frame, lbl, x1, max(y1-8, 14))

            # draw plate label
            if t["plate"]:
                py1b = y2-int((y2-y1)*0.28)
                cv2.rectangle(frame,(x1,py1b),(x2,y2),(0,165,255),2)
                draw_text(frame, t["plate"], x1+4, max(py1b-6,14),
                          scale=0.5, color=(0,220,255))

        # prune stale tracks (not seen for this frame)
        self.tracks = {k:v for k,v in self.tracks.items() if k in seen}

        # HUD
        cv2.line(frame,(0,self.line_y),(self.orig_w,self.line_y),(0,0,220),2)
        cv2.rectangle(frame,(8,8),(210,72),(0,0,0),-1)
        draw_text(frame,f"IN : {self.counts['in']}",  14,30,scale=0.65,color=(100,255,100))
        draw_text(frame,f"OUT: {self.counts['out']}", 14,56,scale=0.65,color=(100,200,255))
        return frame

# ─── Jobs ────────────────────────────────────────────────────────────────────

def run_image_job(job_id, inp, out, enable_ocr):
    try:
        jobs[job_id]["status"] = "running"
        yolo, ocr = get_models(need_ocr=enable_ocr)
        frame = cv2.imread(inp)
        if frame is None: raise ValueError("Cannot read image")
        h,w = frame.shape[:2]
        proc = VideoProcessor(yolo, ocr, w, h, enable_ocr)
        result = proc.process(frame)
        # for image, also try OCR regardless of flag (single frame is fast)
        if enable_ocr:
            pass  # already done above
        cv2.imwrite(out, result)
        jobs[job_id].update({
            "status":"done","progress":100,
            "result": os.path.basename(out),
            "counts": proc.counts,
            "plates": list(proc.all_plates)
        })
    except Exception as e:
        jobs[job_id].update({"status":"error","error":str(e)})

def run_video_job(job_id, inp, out, enable_ocr):
    try:
        jobs[job_id]["status"] = "running"
        yolo, ocr = get_models(need_ocr=enable_ocr)
        cap = cv2.VideoCapture(inp)
        if not cap.isOpened(): raise ValueError("Cannot open video")
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out, fourcc, fps, (W,H))
        proc = VideoProcessor(yolo, ocr, W, H, enable_ocr)
        t0 = time.time()
        while True:
            ret, frame = cap.read()
            if not ret: break
            writer.write(proc.process(frame))
            jobs[job_id]["progress"] = int(proc.fn/total*100) if total else 0
            jobs[job_id]["eta"] = _eta(t0, proc.fn, total)
        cap.release(); writer.release()
        jobs[job_id].update({
            "status":"done","progress":100,
            "result": os.path.basename(out),
            "counts": proc.counts,
            "plates": list(proc.all_plates),
            "eta": "Done"
        })
    except Exception as e:
        jobs[job_id].update({"status":"error","error":str(e)})

def _eta(t0, done, total):
    if done == 0 or total == 0: return "…"
    elapsed = time.time()-t0
    remaining = elapsed/done*(total-done)
    if remaining < 60: return f"{int(remaining)}s left"
    return f"{int(remaining/60)}m {int(remaining%60)}s left"

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file("static/index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error":"No file"}), 400
    f   = request.files["file"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_IMG | ALLOWED_VID:
        return jsonify({"error":"Unsupported file type"}), 400
    enable_ocr = request.form.get("ocr","0") == "1"

    job_id = str(uuid.uuid4())[:8]
    inp    = os.path.join(UPLOAD_FOLDER, f"{job_id}{ext}")
    f.save(inp)
    out_ext = ".jpg" if ext in ALLOWED_IMG else ".mp4"
    out  = os.path.join(OUTPUT_FOLDER, f"{job_id}_out{out_ext}")
    kind = "image" if ext in ALLOWED_IMG else "video"
    jobs[job_id] = {"status":"queued","progress":0,"kind":kind,"eta":"…"}

    fn = run_image_job if kind=="image" else run_video_job
    threading.Thread(target=fn, args=(job_id,inp,out,enable_ocr), daemon=True).start()
    return jsonify({"job_id":job_id,"kind":kind})

@app.route("/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job: return jsonify({"error":"Not found"}),404
    return jsonify(job)

@app.route("/result/<filename>")
def result(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    print("\n🚗  VisionPlate — fast edition")
    print("   Open http://localhost:5000\n")
    app.run(debug=False, port=5000)
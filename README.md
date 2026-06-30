# Object Detection App

A vehicle detection web app powered by YOLOv8 and EasyOCR. Upload an image or video and get back annotated output with bounding boxes, vehicle counts (IN/OUT), speed estimates, and optional license plate recognition.

## Features

- Detects cars, motorcycles, buses, and trucks
- Counts vehicles crossing a line (IN / OUT)
- Estimates speed per vehicle
- Optional license plate OCR
- Supports images (JPG, PNG, BMP, WEBP) and videos (MP4, AVI, MOV, MKV, WMV) up to 200 MB
- Simple drag-and-drop web interface

## Installation

```bash
pip install flask ultralytics easyocr opencv-python numpy
```

## Usage

### Web App

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Command Line

```bash
# Image
python detect.py --input photo.jpg --output result.jpg

# Video
python detect.py --input video.mp4 --output result.mp4
```

## Requirements

- Python 3.10+
- `flask`
- `ultralytics`
- `easyocr`
- `opencv-python`
- `numpy`

The YOLOv8n model (`yolov8n.pt`) is included. EasyOCR models are downloaded automatically on first run.

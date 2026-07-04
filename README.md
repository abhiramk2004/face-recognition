# Real-Time Object Detection, Tracking and Face Recognition

A modular CPU-based computer vision pipeline for real-time **person detection**, **multi-object tracking**, and **face recognition** using **YOLO11n**, **ByteTrack**, and **InsightFace**.

The project is designed for deployment on a **Multi-access Edge Computing (MEC)** server, processing live RTSP streams from IP cameras (including future 5G Standalone cameras).

---

# Features

- Real-time person detection using YOLO11n
- Multi-object tracking using ByteTrack
- Face recognition using InsightFace
- Face enrollment from image datasets
- CPU-only inference
- Webcam support
- RTSP camera support
- Modular software architecture
- Runtime performance monitoring
- Ready for Dockerization
- Designed for MEC deployment

---

# System Architecture

```
                 Camera
          (Webcam / RTSP Stream)

                     │

                     ▼

              Camera Module

                     │

                     ▼

             YOLO11n Detector

                     │

                     ▼

             ByteTrack Tracker

                     │

                     ▼

         Recognition Decision

                     │

                     ▼

          InsightFace Service

                     │

                     ▼

             Visualization

                     │

                     ▼

           Display & Statistics
```

---

# Project Structure

```
mec-vision/
│
├── app/
│   ├── camera/
│   │   └── camera.py
│   │
│   ├── detection/
│   │   ├── detector.py
│   │   └── tracker.py
│   │
│   ├── pipeline/
│   │   └── pipeline.py
│   │
│   ├── recognition/
│   │   ├── enrollment.py
│   │   ├── face_model.py
│   │   ├── recognizer.py
│   │   └── service.py
│   │
│   ├── utils/
│   │   ├── helpers.py
│   │   └── metrics.py
│   │
│   ├── visualization/
│   │   └── renderer.py
│   │
│   ├── config.py
│   ├── main.py
│   └── __init__.py
│
├── faces/
├── models/
│   └── yolo11n.pt
│
├── logs/
│
├── README.md
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

# Module Overview

## Camera

Responsible for

- Webcam capture
- RTSP stream capture
- Frame resizing
- Video source abstraction

---

## Detection

Responsible for

- Loading YOLO11n
- Person detection
- ByteTrack tracking
- Returning tracked objects

---

## Track Manager

Responsible for

- Active track management
- Recognition cache
- Track timeout
- Recognition scheduling

---

## Recognition

Responsible for

- InsightFace initialization
- Face enrollment
- Runtime recognition
- Identity matching

Recognition is executed only

- when a new track appears
- after the configured recheck interval

to reduce CPU utilization.

---

## Visualization

Responsible for

- Bounding boxes
- Labels
- FPS
- Runtime statistics

---

## Metrics

Collects

- FPS
- YOLO inference time
- Face recognition time
- Active track count

---

# Processing Pipeline

```
Frame

↓

YOLO11n Detection

↓

ByteTrack Tracking

↓

Track Manager

↓

Recognition Decision

↓

Face Recognition

↓

Renderer

↓

Display
```

---

# Face Enrollment

Each enrolled identity is stored as

```
faces/

├── Person_A/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...

├── Person_B/
│   └── ...
```

During application startup

- all images are loaded
- embeddings are generated
- embeddings are averaged
- normalized embeddings are stored in memory

---

# Configuration

Application parameters are centralized in

```
app/config.py
```

Common configurable parameters include

| Parameter | Description |
|------------|-------------|
| USE_RTSP | Select RTSP or webcam |
| RTSP_URL | RTSP camera URL |
| WEBCAM_INDEX | Webcam index |
| FRAME_WIDTH | Processing width |
| FRAME_HEIGHT | Processing height |
| YOLO_MODEL | YOLO model path |
| DEVICE | CPU / GPU |
| PERSON_CLASS_ID | Detection class |
| TRACK_TIMEOUT | Track timeout |
| RECHECK_INTERVAL | Face recognition interval |
| SIMILARITY_THRESHOLD | Face matching threshold |
| FACE_DATABASE_PATH | Face dataset directory |

---

# Current Performance

Current development configuration

- CPU-only inference
- YOLO11n
- ByteTrack
- InsightFace
- Processing resolution: **704 × 480**

Initial implementation

- YOLO inference: ~35–40 ms
- Tracking: ~24–25 FPS

Current modular implementation

- Functionally complete
- Undergoing performance optimization after modular refactoring

---

# Running

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python -m app.main
```

---

# Current Status

Completed

- ✅ Modular architecture
- ✅ Webcam support
- ✅ RTSP support
- ✅ Person detection
- ✅ ByteTrack tracking
- ✅ Face recognition
- ✅ Runtime statistics
- ✅ Project refactoring

Current focus

- Performance optimization
- Dockerization
- MEC deployment

---

# Known Limitations

## RTSP Latency

The current implementation uses OpenCV's `VideoCapture()` for RTSP reception.

Observed latency

| Method | Approximate Latency |
|---------|---------------------|
| OpenCV VideoCapture | ~1 second |
| FFplay | <0.5 second |

This suggests buffering overhead within the current OpenCV capture pipeline.

---

## Sequential Processing

The current processing pipeline is sequential.

```
Capture

↓

YOLO

↓

Tracking

↓

Recognition

↓

Rendering
```

If processing becomes slower than the incoming frame rate, latency accumulates because every frame is processed in order.

---

# Planned Performance Optimizations

## Camera Thread

Separate frame acquisition from inference.

```
RTSP

↓

Camera Thread

↓

Latest Frame Buffer
```

This prevents object detection from blocking frame reception.

---

## Frame Dropping

Instead of processing every incoming frame

```
Frame1
Frame2
Frame3
Frame4
Frame5
```

the system will always process the newest available frame

```
Frame5
```

This minimizes end-to-end latency and maintains real-time responsiveness.

---

## Queue-Based Pipeline

Introduce a bounded queue

```
Queue(maxsize=1)
```

Architecture

```
Camera

↓

Queue

↓

YOLO
```

Old frames will be discarded automatically, preventing queue buildup.

---

## FFmpeg-Based RTSP Capture

Replace OpenCV's RTSP receiver with an FFmpeg-based capture backend using low-latency options.

Expected improvements

- Lower buffering
- Reduced latency
- Better live synchronization

---

## Asynchronous Recognition

Recognition will be decoupled from detection.

```
YOLO

↓

Immediate Display

↓

Recognition Thread

↓

Identity Update
```

Detection continues while recognition executes independently.

---

## Multiprocessing

Future architecture

```
Camera Process

↓

Shared Frame Buffer

↓

Detection Process

↓

Recognition Process

↓

Renderer
```

This allows better CPU utilization on multicore systems.

---

# Technologies Used

- Python
- OpenCV
- Ultralytics YOLO11
- ByteTrack
- InsightFace
- NumPy

---

# Version

Current Version: **v0.1.0**

This release provides a fully modular CPU-based vision pipeline supporting real-time person detection, tracking, and face recognition, and serves as the baseline before Dockerization and latency optimization.

---


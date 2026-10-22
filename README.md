# Multi-Vehicle Speed Detection System

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, standalone **Multi-Vehicle Speed Detection System** in Python using **YOLOv8** object detection and a custom **Centroid Tracker** built from scratch.

This system detects multiple vehicles (Cars, Motorcycles, Buses, Trucks) in real-time from fixed CCTV traffic video streams, tracks them across frames with persistent IDs, measures the precise frame time taken to travel between two virtual reference lines, and calculates speed in **km/h**.

---

## 🌟 Key Features

- **YOLOv8 Object Detection**: Pre-trained model (`yolov8n.pt`) automatically detects vehicle classes (`Car`, `Motorcycle`, `Bus`, `Truck`).
- **Custom Centroid Tracker**: Built from scratch using Euclidean distance matching without external tracking libraries.
- **Two-Line Virtual Speed Trap**:
  - Entry Line (Yellow) records start time.
  - Exit Line (Red) records stop time.
  - Speed formula: $\text{Speed (km/h)} = \frac{\text{Distance (m)}}{\Delta t \text{ (s)}} \times 3.6$.
- **Real-Time HUD Overlay**:
  - Semi-transparent stats dashboard (vehicles tracked, in zone, average/max speed, FPS).
  - Vehicle classification breakdown bar.
  - Motion trail tails behind tracked vehicles.
  - Over-speed alert highlighting (Red bounding box & ⚠️ indicator).
- **Automated Overspeed Snapshot**: Saves high-speed vehicle crops directly to `output/overspeeding/`.
- **CSV Speed Logging**: Automatic event logging with timestamps to `logs/speed_log_YYYY-MM-DD.csv`.
- **Speed Histogram Generator**: Saves a matplotlib speed distribution graph (`output/speed_histogram.png`) upon completion.
- **Exhibition Live Controls**: Adjust line positions live during playback using keys `1`, `2`, `3`, `4`.

---

## 📁 Project Structure

```
Project-Exhibition-1/
├── speed_detector.py          # Main application & processing pipeline
├── tracker.py                 # Pure-Python CentroidTracker class
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
├── PROJECT_REPORT.md          # Technical report & system architecture
├── LICENSE                    # MIT Open-Source License
├── videos/                    # Input traffic sample videos
├── output/                    # Generated output videos, screenshots & charts
│   ├── overspeeding/          # Saved crops of speeding vehicles
│   └── speed_histogram.png    # Session speed distribution chart
└── logs/                      # CSV speed log entries
```

---

## ⚡ Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ISHANT-AG/Project-Exhibition-1.git
   cd Project-Exhibition-1
   ```

2. **Create and Activate a Virtual Environment** (Optional but recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
   *(Note: `yolov8n.pt` auto-downloads on first run (~6MB))*

---

## 🚀 How to Run

### Basic Usage:
Run the detector with the included sample video:
```bash
python3 speed_detector.py --video videos/sample_traffic.mp4
```

### Custom Settings:
```bash
python3 speed_detector.py --video videos/sample_traffic.mp4 --distance 15.0 --speed-limit 50 --output output/result_demo.mp4
```

### CLI Arguments:
- `--video`: Path to input video file (required)
- `--orientation`: Line orientation: `horizontal` (default) or `vertical`
- `--distance`: Real-world distance between reference lines in meters (default: `15.0`)
- `--speed-limit`: Speed limit threshold for violations in km/h (default: `60.0`)
- `--conf`: YOLO detection confidence threshold (default: `0.35`)
- `--output`: Path to save annotated output video

---

## ⌨️ Live Keyboard Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / Resume playback |
| `Q` or `ESC` | Quit session and display final summary report |
| `S` | Save full-frame screenshot to `output/` |
| `+` / `-` | Increase / Decrease playback speed |
| `O` | Toggle Reference Line Orientation (Horizontal / Vertical) |
| `1` | Move Entry Line **UP / LEFT** |
| `2` | Move Entry Line **DOWN / RIGHT** |
| `3` | Move Exit Line **UP / LEFT** |
| `4` | Move Exit Line **DOWN / RIGHT** |

---

## 📐 Calibration Guide

1. **Identify Road References**: Find two distinct road points (e.g. lane divider segments or poles) visible in your video. Standard highway dashed lane markings are typically 3.0m dash with 4.5–9m gap.
2. **Estimate Distance**: Count how many markings span between Line 1 and Line 2, and pass `--distance <meters>`.
3. **Live Fine-Tuning**: Use keys `1/2` and `3/4` while the video plays to position the virtual lines perpendicular to traffic flow.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

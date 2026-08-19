# Multi-Vehicle Speed Detection System

A complete, standalone **Multi-Vehicle Speed Detection System** in Python using **YOLOv8** object detection and a custom **Centroid Tracker** built from scratch.

This system detects multiple vehicles (Cars, Motorcycles, Buses, Trucks) in real-time from a fixed CCTV traffic video stream, tracks them across frames with persistent IDs, measures the precise frame time taken to travel between two virtual reference lines, and calculates speed in **km/h**.

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
Project Exhibition - 2/
├── speed_detector.py          # Main application & processing pipeline
├── tracker.py                 # Pure-Python CentroidTracker class
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
├── Vehicle_Speed_Detection_Prompt.md
├── videos/                    # Input traffic sample videos
├── output/                    # Generated output videos, screenshots & charts
│   ├── overspeeding/          # Saved crops of speeding vehicles
│   └── speed_histogram.png    # Session speed distribution chart
└── logs/                      # CSV speed log entries
```

---

## ⚡ Installation & Setup

1. **Clone or Navigate to Project Directory**:
   ```bash
   cd "Project Exhibition - 2"
   ```

2. **Install Dependencies**:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
   *(Note: `yolov8n.pt` auto-downloads on first run (~6MB))*

---

## 🚀 How to Run

### Basic Usage:
Place a sample traffic video in the `videos/` folder and run:
```bash
python3 speed_detector.py --video videos/sample_traffic.mp4
```

### Custom Settings:
```bash
python3 speed_detector.py --video videos/sample_traffic.mp4 --distance 15.0 --speed-limit 50 --output output/result_demo.mp4
```

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

## 📐 Calibration Guide for Exhibition

1. **Identify Road References**: Find two distinct road points (e.g. lane divider segments or poles) visible in your video. In India, standard dashed lane markings are ~3.0m with ~4.5m gaps (7.5m cycle).
2. **Estimate Distance**: Count how many markings span between Line 1 and Line 2, and pass `--distance <meters>`.
3. **Live Fine-Tuning**: Use keys `1/2` and `3/4` while the video plays to position the virtual lines perpendicular to traffic flow.
# Project-Exhibition-1

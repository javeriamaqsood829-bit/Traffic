# Multi-Vehicle Speed Detection System (V2 Production Suite)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8n-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-5.x-red.svg)](https://opencv.org/)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack%20(Kalman%2BIoU)-orange.svg)](tracker.py)
[![ALPR](https://img.shields.io/badge/ALPR-EasyOCR-purple.svg)](alpr.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-grade **AI Multi-Vehicle Speed Detection & Traffic Enforcement System** in Python combining **YOLOv8**, **ByteTrack (Kalman Filter + IoU)**, **Inverse Perspective Mapping (IPM Homography)**, **ALPR Automatic License Plate Recognition**, **Night Vision Enhancement**, and **SQLite State Persistence**.

---

## 🌟 The 6 Next-Gen Technologies Implemented

### 1. 📋 Automatic License Plate Recognition (ALPR) & E-Challan Generator (`alpr.py`)
- Automatically isolates violation vehicle crops and isolates plate candidate regions using morphological black-hat and Sobel gradient filtering.
- Performs alphanumeric character extraction using **EasyOCR** with regex validation for national vehicle registration syntaxes.
- Generates official, high-resolution **Traffic Violation Citation (E-Challan)** evidence cards stamped with detected plate, violation speed, excess margin, timestamp, and camera metadata saved to `output/overspeeding/`.

### 2. 📐 4-Point Homography & Inverse Perspective Mapping (`homography.py`)
- Resolves camera pitch and oblique angle distortion by computing the $3 \times 3$ Homography transform matrix $H$ between image plane $(u, v)$ and physical ground coordinates $(X, Y)$ in real-world **meters**.
- Computes true Euclidean ground speed: $\text{Speed} = \frac{\sqrt{\Delta X^2 + \Delta Y^2}}{\Delta t} \times 3.6$ km/h.
- Generates a live, top-down **Bird's-Eye View IPM Radar** inset on the HUD displaying physical vehicle positions.

### 3. 🌙 Night Vision / Low-Light CLAHE & Headlight Glare Suppression (`enhancer.py`)
- Auto-detects night or low-contrast footage via frame luminance analysis ($V$-channel).
- Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) on the luminance channel in LAB color space to recover dark vehicle silhouettes.
- Employs **Headlight Glare Suppression** to soften specular headlight blooms and prevent bounding box ballooning or centroid drift.

### 4. 🎯 ByteTrack Multi-Object Tracker (`tracker.py`)
- State-of-the-art tracker using **Kalman Filtering** (8-state vector: $[cx, cy, s, r, \dot{cx}, \dot{cy}, \dot{s}, \dot{r}]$) with constant-velocity prediction across missing frames.
- **Two-Stage IoU Association** with the Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) to seamlessly match both high- and low-confidence detections, eliminating ID switches and occlusion dropouts.
- Backwards-compatible fallback to pure-Python `CentroidTracker`.

### 5. 🛺 Regional Vehicle Classifier & Multi-Lane Assignment (`classifier.py`)
- Extends standard COCO classes (`Car`, `Motorcycle`, `Bus`, `Truck`) with heuristic detection for regional vehicles such as **Auto-Rickshaws (Three-Wheelers)** based on aspect ratio and livery color profiling.
- Distinguishes **Light Commercial Vehicles (LCV / Vans)** from Heavy Trucks.
- Computes real-time **Spatial Lane Assignment** (e.g. Lane 1 Left, Lane 2 Center, Lane 3 Right).

### 6. 🗄️ Decoupled SQLite State Persistence & 4-Panel Analytics (`database.py`)
- Decouples ephemeral tracking state from immutable audit persistence using SQLite (`logs/violations.db`) with WAL concurrency.
- Dual-sync exports: clean CSV logs (`logs/speed_log_YYYY-MM-DD.csv`) and REST-ready JSON summaries (`logs/session_summary.json`).
- Automatically generates a comprehensive 4-panel visual report (`output/session_analytics.png`):
  1. Speed Distribution Histogram with Speed Limit and Mean markers.
  2. Vehicle Classification Breakdown Bar Chart.
  3. Speed Compliance Donut Chart.
  4. Sequential Passage Speed Timeline.

---

## 📁 Project Architecture

```
Project-Exhibition-1/
├── speed_detector.py          # Unified pipeline runner & HUD orchestrator
├── tracker.py                 # ByteTrack (Kalman + IoU) & CentroidTracker
├── alpr.py                    # License Plate Recognition & E-Challan citations
├── homography.py              # 4-point IPM perspective calibration & bird's-eye radar
├── enhancer.py                # Night CLAHE enhancement & headlight glare suppression
├── classifier.py              # Regional auto-rickshaw & lane assignment engine
├── database.py                # SQLite persistence, CSV/JSON export & analytics charts
├── requirements.txt           # Python dependencies
├── README.md                  # System documentation
├── PROJECT_REPORT.md          # In-depth technical architecture report
├── videos/                    # Sample traffic surveillance footage
├── output/                    # Annotated videos, screenshots & charts
│   ├── overspeeding/          # Generated citation cards and vehicle evidence
│   ├── session_analytics.png  # 4-panel session analytics report
│   └── speed_histogram.png    # Session speed distribution chart
└── logs/                      # SQLite DB, CSV logs & JSON summaries
    ├── violations.db          # SQLite persistent violation database
    ├── speed_log_YYYY-MM-DD.csv # Tabular passage log
    └── session_summary.json   # Machine-readable summary export
```

---

## ⚡ Installation & Quick Start

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/ISHANT-AG/Project-Exhibition-1.git
   cd Project-Exhibition-1
   ```

2. **Set up Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. **Run the System**:
   ```bash
   # Run with ByteTrack + ALPR E-Challans + SQLite DB
   python3 speed_detector.py --video videos/sample_traffic.mp4

   # Run with 4-Point IPM Homography Ground Calibration
   python3 speed_detector.py --video videos/sample_traffic.mp4 --homography

   # Run with Night Mode CLAHE & Glare Suppression
   python3 speed_detector.py --video videos/sample_traffic.mp4 --night-mode

   # Run headlessly (e.g. on server / cloud pipeline)
   python3 speed_detector.py --video videos/sample_traffic.mp4 --no-show
   ```

---

## ⌨️ Live Keyboard Controls

| Key | Function |
|---|---|
| `SPACE` | Pause / Resume video playback |
| `Q` or `ESC` | Quit session and export final analytics report |
| `T` | Toggle Tracker Core (`ByteTrack (Kalman+IoU)` $\leftrightarrow$ `Centroid`) |
| `H` | Toggle 4-Point Homography Ground Calibration Zone & IPM Radar |
| `N` | Toggle Night Vision / Low-Light CLAHE Enhancement & Glare Filter |
| `O` | Toggle Reference Line Orientation (`Horizontal` $\leftrightarrow$ `Vertical`) |
| `S` | Save full-frame screenshot to `output/` |
| `1` / `2` | Move Entry Line UP / DOWN (or LEFT / RIGHT) |
| `3` / `4` | Move Exit Line UP / DOWN (or LEFT / RIGHT) |

---

## 📊 Sample Output & Audit Trail

- **Evidence Citation Cards**: Saved automatically to `output/overspeeding/citation_id<ID>_<SPEED>kmh_<PLATE>.jpg`
- **SQLite Database**: Queryable anytime using `sqlite3 logs/violations.db "SELECT * FROM violations;"`
- **Visual Analytics**: Multi-panel report saved to `output/session_analytics.png`

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

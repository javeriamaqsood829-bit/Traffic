# ============================================================
# Multi-Vehicle Speed Detection System (V2 — Production Suite)
# ============================================================
# Core Engine integrating 6 Next-Gen Technologies:
#   1. Automatic License Plate Recognition (ALPR / E-Challan)
#   2. 4-Point Homography & Inverse Perspective Mapping (IPM)
#   3. Low-Light / Night Enhancement & Headlight Glare Suppression
#   4. ByteTrack-style Kalman Filter + IoU Multi-Object Tracker
#   5. Fine-Grained & Regional Vehicle Classification (Auto-Rickshaw, LCV)
#   6. Decoupled SQLite State Persistence & Multi-Panel Analytics
# ============================================================

import os
import sys
import time
import math
import argparse
import datetime
import csv
from collections import defaultdict, deque
import cv2
import numpy as np

# Import 6 Modular Subsystems
from tracker import get_tracker, ByteTrackTracker, CentroidTracker
from alpr import ALPREngine, generate_violation_citation
from homography import PerspectiveCalibrator
from enhancer import NightEnhancer
from classifier import VehicleClassifier
from database import ViolationDatabase, generate_session_analytics

# Try importing ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' module not found. Please install it using: python3 -m pip install ultralytics")
    sys.exit(1)

# ===== DEFAULT CONFIGURATION =====
MODEL_NAME = "yolov8n.pt"            # YOLOv8 nano model (fast & lightweight)
CONFIDENCE_THRESHOLD = 0.35          # Minimum detection confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]        # COCO IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck

LINE_1_POSITION = 0.35               # Entry line default (35% mark)
LINE_2_POSITION = 0.65               # Exit line default (65% mark)
DEFAULT_ORIENTATION = "horizontal"   # Default orientation ("horizontal" or "vertical")
KNOWN_DISTANCE_METERS = 15.0         # Reference distance in meters

MAX_DISAPPEARED_FRAMES = 40          # Frames before lost vehicle is removed
MAX_MATCH_DISTANCE = 80              # Max pixel distance for centroid matching

SPEED_LIMIT_KMH = 60.0               # Speed limit threshold in km/h
MIN_REASONABLE_SPEED = 5.0           # Ignore lower values (stationary/noise)
MAX_REASONABLE_SPEED = 200.0         # Ignore higher values (tracking glitch)

# Display Colors (BGR format)
LINE_COLOR_1 = (0, 255, 255)         # Yellow for Entry Line
LINE_COLOR_2 = (0, 0, 255)           # Red for Exit Line
SAFE_SPEED_COLOR = (0, 255, 0)       # Green for normal speed
OVER_SPEED_COLOR = (0, 0, 255)       # Red for overspeeding
TEXT_COLOR = (255, 255, 255)         # White text
PANEL_BG_COLOR = (20, 20, 24)        # Dark slate background for HUD


def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_length=15, gap_length=10):
    """Draw a dashed line across the frame between pt1 and pt2."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist == 0:
        return

    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    curr_dist = 0
    draw = True

    while curr_dist < dist:
        segment_len = dash_length if draw else gap_length
        next_dist = min(curr_dist + segment_len, dist)
        start_pt = (int(x1 + dx * curr_dist), int(y1 + dy * curr_dist))
        end_pt = (int(x1 + dx * next_dist), int(y1 + dy * next_dist))
        if draw:
            cv2.line(img, start_pt, end_pt, color, thickness)
        curr_dist = next_dist
        draw = not draw


def draw_label(frame, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.52, thickness=1):
    """Draw text with a solid filled background box for high readability."""
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    x = max(2, min(frame.shape[1] - text_w - 8, x))
    y = max(text_h + baseline + 6, y)
    cv2.rectangle(frame, (x, y - text_h - baseline - 4), (x + text_w + 6, y + 2), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


def detect_vehicles(model, frame, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Run YOLOv8 object detection on a single frame and extract vehicle bounding boxes.
    """
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detections = []
    classes = []
    confidences = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2))
            classes.append(cls_id)
            confidences.append(conf)

    return detections, classes, confidences


def draw_dashboard(frame, stats):
    """
    Draw an upgraded, multi-system HUD dashboard panel in the top-left corner.
    """
    panel_w = 360
    panel_h = 275
    overlay = frame.copy()

    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), PANEL_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (90, 90, 110), 1)

    # Title Banner
    cv2.putText(frame, "AI SPEED ENFORCEMENT V2", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 42), (10 + panel_w - 10, 42), (100, 100, 120), 1)

    night_badge = "ACTIVE" if stats["night_mode"] else "OFF"
    ipm_badge = "CALIBRATED (IPM)" if stats["homography_mode"] else "2-LINE TRAP"
    tracker_badge = stats["tracker_type"].upper()

    lines = [
        f"Tracker Core: {tracker_badge} (Kalman+IoU)" if "BYTE" in tracker_badge else f"Tracker Core: {tracker_badge}",
        f"Night Vision: {night_badge} (CLAHE + Glare)",
        f"Speed Calibration: {ipm_badge}",
        f"ALPR Module: {'ENABLED (OCR)' if stats['alpr_enabled'] else 'DISABLED'}",
        f"Vehicles Tracked: {stats['tracked_total']}  |  In Zone: {stats['in_zone']}",
        f"Speeds Measured: {stats['speeds_count']}",
        f"Avg: {stats['avg_speed']:.1f} km/h  |  Max: {stats['max_speed']:.1f} km/h" if stats['speeds_count'] > 0 else "Avg: -- km/h  |  Max: -- km/h",
        f"Over Speed Violations: {stats['over_limit_count']}",
        f"System FPS: {stats['fps']:.1f}"
    ]

    y = 66
    for line in lines:
        color = TEXT_COLOR
        if "Over Speed Violations" in line and stats['over_limit_count'] > 0:
            color = (0, 0, 255)
        elif "Tracker Core" in line:
            color = (0, 230, 255)
        elif "Night Vision" in line and stats["night_mode"]:
            color = (100, 255, 100)
        elif "Speed Calibration" in line:
            color = (255, 200, 80)
        elif "System FPS" in line:
            color = (180, 180, 180)

        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
        y += 22


def draw_vehicle_count_bar(frame, counts):
    """
    Draw extended vehicle classification counts panel on top-right.
    """
    h, w = frame.shape[:2]
    panel_w = 210
    panel_h = 160
    x_start = w - panel_w - 10
    y_start = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), PANEL_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (90, 90, 110), 1)

    cv2.putText(frame, "VEHICLE CLASSIFIER", (x_start + 12, y_start + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

    y = y_start + 48
    all_classes = ["Car", "Motorcycle", "Bus", "Truck", "Auto-Rickshaw", "Commercial"]
    for cls_name in all_classes:
        count = counts.get(cls_name, 0)
        color = VehicleClassifier.CLASS_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(frame, (x_start + 12, y - 9), (x_start + 22, y + 1), color, cv2.FILLED)
        cv2.putText(frame, f"{cls_name}: {count}", (x_start + 28, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 18


def process_video(args):
    """Main video processing pipeline integrating all 6 technologies."""
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found. Please check the file path.")
        return

    # Ensure required directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/overspeeding", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("\n" + "═" * 65)
    print(" 🚗 MULTI-VEHICLE SPEED DETECTION SYSTEM — V2 PRODUCTION SUITE")
    print("═" * 65)
    print(f" Initializing YOLOv8 model '{MODEL_NAME}'...")

    try:
        model = YOLO(MODEL_NAME)
        print(" YOLOv8 loaded successfully! ✅")
    except Exception as e:
        print(f"❌ Error downloading/loading YOLO model: {e}")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: OpenCV cannot open video file '{video_path}'.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    orientation = args.orientation.lower()

    # Initialize 6 Core Technologies
    print(f" [1/6] ALPR & Citation Generator: {'Enabled (EasyOCR)' if args.alpr else 'Disabled'}")
    alpr_engine = ALPREngine(use_ocr=args.alpr)

    print(f" [2/6] Homography Calibrator: {'Enabled (IPM Mode)' if args.homography else '2-Line Virtual Trap'}")
    calibrator = PerspectiveCalibrator(road_width_m=8.0, road_length_m=35.0)
    calibrator.set_default_preset(frame_w, frame_h)

    print(f" [3/6] Night Enhancer: {'Forced ON' if args.night_mode else 'Auto-detecting Low Light'}")
    enhancer = NightEnhancer(clip_limit=2.5, glare_suppression=True)

    tracker_type = args.tracker.lower()
    print(f" [4/6] Multi-Object Tracker: {tracker_type.upper()} (Kalman Filter + IoU)")
    tracker = get_tracker(tracker_type=tracker_type, max_disappeared=MAX_DISAPPEARED_FRAMES)

    print(" [5/6] Vehicle Classifier: Regional Auto-Rickshaw + Multi-Lane Engine")
    classifier = VehicleClassifier(num_lanes=3)

    print(" [6/6] State Persistence: SQLite Audit Database ('logs/violations.db')")
    db = ViolationDatabase(args.db)

    print("-" * 65)
    print(f" Video Source: {video_path} ({frame_w}x{frame_h} @ {fps:.2f} FPS)")
    print(f" Speed Limit: {args.speed_limit:.1f} km/h | Trap Distance: {args.distance:.1f} m")
    print(" Controls: [SPACE] Pause | [Q] Quit | [N] Night Mode | [H] Homography/IPM | [T] Switch Tracker")
    print("-" * 65 + "\n")

    # Output video writer
    output_path = args.output
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join("output", f"result_{base_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not out_writer.isOpened():
        output_path = os.path.splitext(output_path)[0] + ".avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    # Tracking states
    line1_ratio = LINE_1_POSITION
    line2_ratio = LINE_2_POSITION

    line1_cross_time = {}
    line2_cross_time = {}
    vehicle_speeds = {}
    vehicle_meta = {}
    crossed_line1 = set()
    crossed_line2 = set()
    overspeed_saved = set()
    vehicle_trails = defaultdict(lambda: deque(maxlen=30))
    class_counts = defaultdict(int)

    # Runtime toggles
    night_mode_active = args.night_mode
    homography_view = args.homography

    frame_idx = 0
    paused = False
    delay_ms = int(1000 / fps) if fps > 0 else 30

    last_fps_time = time.time()
    fps_counter = 0
    current_fps = fps

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            fps_counter += 1

        # Use hardware container presentation timestamp (PTS) if available; fallback to frame_idx / fps
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        current_video_time = (pos_msec / 1000.0) if (pos_msec is not None and pos_msec > 0) else (frame_idx / fps)

        # --- STEP 1: NIGHT / LOW-LIGHT ENHANCEMENT ---
        enhanced_frame, is_enhanced = enhancer.enhance(frame, force_night=night_mode_active, auto_detect=not night_mode_active)
        effective_frame = enhanced_frame if (night_mode_active or is_enhanced) else frame

        # --- STEP 2: YOLO DETECTION ---
        raw_detections, raw_classes, raw_confs = detect_vehicles(model, effective_frame, conf_threshold=args.conf)

        # --- STEP 3: MULTI-OBJECT TRACKING ---
        tracked_objects = tracker.update(raw_detections)

        # --- STEP 4: REGIONAL CLASSIFICATION & METADATA ---
        for rect, cls_id, conf in zip(raw_detections, raw_classes, raw_confs):
            rcx = int((rect[0] + rect[2]) / 2.0)
            rcy = int((rect[1] + rect[3]) / 2.0)
            for vid, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
                if math.hypot(cx - rcx, cy - rcy) < 35:
                    if vid not in vehicle_meta:
                        crop_car = frame[max(0, y1):min(frame_h, y2), max(0, x1):min(frame_w, x2)]
                        meta = classifier.classify(crop_car, cls_id, conf, (x1, y1, x2, y2), frame_w)
                        vehicle_meta[vid] = meta
                        class_counts[meta["class_name"]] += 1

        # Calculate Line positions
        if orientation == "horizontal":
            line1_pos = int(frame_h * line1_ratio)
            line2_pos = int(frame_h * line2_ratio)
        else:
            line1_pos = int(frame_w * line1_ratio)
            line2_pos = int(frame_w * line2_ratio)

        entry_line = min(line1_pos, line2_pos)
        exit_line = max(line1_pos, line2_pos)

        # --- STEP 5: SPEED MEASUREMENT & CITATION GENERATION ---
        in_zone_count = 0
        for vehicle_id, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
            has_history = len(vehicle_trails[vehicle_id]) > 0
            prev_pt = vehicle_trails[vehicle_id][-1] if has_history else (cx, cy)
            vehicle_trails[vehicle_id].append((cx, cy))

            curr_coord = cy if orientation == "horizontal" else cx
            prev_coord = prev_pt[1] if orientation == "horizontal" else prev_pt[0]

            if entry_line <= curr_coord <= exit_line:
                in_zone_count += 1

            # Line crossing logic
            if has_history:
                # Forward motion
                if prev_coord <= entry_line < curr_coord and vehicle_id not in line1_cross_time:
                    line1_cross_time[vehicle_id] = (current_video_time, (cx, cy))
                    crossed_line1.add(vehicle_id)
                elif prev_coord <= exit_line < curr_coord and vehicle_id not in line2_cross_time:
                    line2_cross_time[vehicle_id] = (current_video_time, (cx, cy))
                    crossed_line2.add(vehicle_id)

                # Reverse motion
                elif curr_coord <= exit_line < prev_coord and vehicle_id not in line2_cross_time:
                    line2_cross_time[vehicle_id] = (current_video_time, (cx, cy))
                    crossed_line2.add(vehicle_id)
                elif curr_coord <= entry_line < prev_coord and vehicle_id not in line1_cross_time:
                    line1_cross_time[vehicle_id] = (current_video_time, (cx, cy))
                    crossed_line1.add(vehicle_id)

            # Speed calculation when both lines are crossed
            if vehicle_id in line1_cross_time and vehicle_id in line2_cross_time and vehicle_id not in vehicle_speeds:
                t1, pt1 = line1_cross_time[vehicle_id]
                t2, pt2 = line2_cross_time[vehicle_id]
                dt = abs(t2 - t1)

                if dt > 0.05:
                    if homography_view:
                        # Use IPM Ground Calibration
                        speed_kmh = calibrator.calculate_speed_ipm(pt1, t1, pt2, t2)
                        if speed_kmh is None:
                            speed_kmh = (args.distance / dt) * 3.6
                    else:
                        # Standard 2-Line Virtual Trap
                        speed_kmh = (args.distance / dt) * 3.6

                    if MIN_REASONABLE_SPEED <= speed_kmh <= MAX_REASONABLE_SPEED:
                        vehicle_speeds[vehicle_id] = round(speed_kmh, 1)
                        meta = vehicle_meta.get(vehicle_id, {"class_name": "Vehicle", "lane_id": "Lane 1"})
                        v_class = meta["class_name"]
                        lane = meta["lane_id"]
                        time_stamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Log passage to SQLite DB
                        db.log_passage(time_stamp_now, vehicle_id, v_class, speed_kmh, args.speed_limit, lane)

                        over_tag = " [SPEED VIOLATION! ⚠️]" if speed_kmh > args.speed_limit else ""
                        print(f"⚡ [EVENT] Vehicle #{vehicle_id:<3} ({v_class:<13}) | Speed: {speed_kmh:.1f} km/h | {lane}{over_tag}")

                        # Handle overspeeding violation
                        if speed_kmh > args.speed_limit and vehicle_id not in overspeed_saved:
                            overspeed_saved.add(vehicle_id)
                            try:
                                crop_x1 = max(0, x1 - 10)
                                crop_y1 = max(0, y1 - 10)
                                crop_x2 = min(frame_w, x2 + 10)
                                crop_y2 = min(frame_h, y2 + 10)
                                vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                                plate_str, plate_conf = "N/A", 0.0
                                if args.alpr and vehicle_crop.size > 0:
                                    plate_str, plate_conf, _, _ = alpr_engine.recognize_plate(vehicle_crop, vehicle_id)

                                # Generate and save citation card
                                citation_meta = {
                                    "vehicle_id": vehicle_id,
                                    "vehicle_class": v_class,
                                    "speed_kmh": speed_kmh,
                                    "speed_limit": args.speed_limit,
                                    "timestamp": time_stamp_now
                                }
                                citation_path = generate_violation_citation(
                                    vehicle_crop, plate_str, plate_conf, citation_meta
                                )

                                # Log violation to SQLite Database
                                db.log_violation({
                                    "timestamp": time_stamp_now,
                                    "vehicle_id": vehicle_id,
                                    "vehicle_class": v_class,
                                    "speed_kmh": speed_kmh,
                                    "speed_limit": args.speed_limit,
                                    "plate_number": plate_str,
                                    "plate_confidence": plate_conf,
                                    "evidence_path": citation_path,
                                    "lane": lane,
                                    "camera_id": "CAM-01"
                                })

                                if plate_str != "N/A":
                                    print(f"  └── 📋 E-Challan Issued! Plate: [{plate_str}] (Conf: {plate_conf*100:.1f}%) -> {citation_path}")
                            except Exception as ex:
                                print(f"Warning: Citation generation notice: {ex}")

        # Update FPS
        now = time.time()
        if now - last_fps_time >= 1.0:
            current_fps = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        # --- STEP 6: DRAW OVERLAYS & HUD ---
        display_frame = effective_frame.copy()

        # Homography ground calibration trapezoid overlay
        if homography_view:
            calibrator.draw_calibration_zone(display_frame)

        # Draw entry and exit lines
        if orientation == "horizontal":
            draw_dashed_line(display_frame, (0, line1_pos), (frame_w, line1_pos), LINE_COLOR_1, thickness=2)
            draw_label(display_frame, f"ENTRY LINE - {int(line1_ratio * 100)}%", 15, line1_pos - 5, LINE_COLOR_1, (0, 0, 0))

            draw_dashed_line(display_frame, (0, line2_pos), (frame_w, line2_pos), LINE_COLOR_2, thickness=2)
            draw_label(display_frame, f"EXIT LINE - {int(line2_ratio * 100)}%", 15, line2_pos - 5, LINE_COLOR_2, (255, 255, 255))
        else:
            draw_dashed_line(display_frame, (line1_pos, 0), (line1_pos, frame_h), LINE_COLOR_1, thickness=2)
            draw_label(display_frame, f"ENTRY LINE - {int(line1_ratio * 100)}%", line1_pos + 5, 260, LINE_COLOR_1, (0, 0, 0))
            draw_dashed_line(display_frame, (line2_pos, 0), (line2_pos, frame_h), LINE_COLOR_2, thickness=2)
            draw_label(display_frame, f"EXIT LINE - {int(line2_ratio * 100)}%", line2_pos + 5, 300, LINE_COLOR_2, (255, 255, 255))

        # Draw vehicle tracking boxes and motion trails
        for vehicle_id, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
            meta = vehicle_meta.get(vehicle_id, {"class_name": "Vehicle", "color": (255, 255, 255)})
            v_class = meta["class_name"]

            if vehicle_id in vehicle_speeds:
                spd = vehicle_speeds[vehicle_id]
                box_color = OVER_SPEED_COLOR if spd > args.speed_limit else SAFE_SPEED_COLOR
                speed_str = f"{spd:.1f} km/h"
                if spd > args.speed_limit:
                    speed_str += " ⚠️"
            elif vehicle_id in crossed_line1 or vehicle_id in crossed_line2:
                box_color = (0, 255, 255)
                speed_str = "Measuring..."
            else:
                box_color = meta["color"]
                speed_str = ""

            # Motion trail tail
            trail_pts = list(vehicle_trails[vehicle_id])
            for i in range(1, len(trail_pts)):
                cv2.line(display_frame, trail_pts[i - 1], trail_pts[i], box_color, 2)

            # Bounding box & center circle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(display_frame, (cx, cy), 4, box_color, -1)

            # Label text
            label_text = f"#{vehicle_id} {v_class}"
            if speed_str:
                label_text += f" | {speed_str}"
            draw_label(display_frame, label_text, x1, y1 - 5, box_color, (255, 255, 255) if box_color != LINE_COLOR_1 else (0, 0, 0))

        # Dashboard stats
        captured_speeds = list(vehicle_speeds.values())
        speeds_count = len(captured_speeds)
        avg_speed = float(np.mean(captured_speeds)) if speeds_count > 0 else 0.0
        max_speed = float(np.max(captured_speeds)) if speeds_count > 0 else 0.0
        over_limit_count = sum(1 for s in captured_speeds if s > args.speed_limit)

        dashboard_stats = {
            "tracker_type": tracker_type,
            "night_mode": night_mode_active or is_enhanced,
            "homography_mode": homography_view,
            "alpr_enabled": args.alpr,
            "tracked_total": tracker.next_object_id,
            "in_zone": in_zone_count,
            "speeds_count": speeds_count,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "over_limit_count": over_limit_count,
            "fps": current_fps
        }

        draw_dashboard(display_frame, dashboard_stats)
        draw_vehicle_count_bar(display_frame, class_counts)

        # Inset Top-Down Bird's-Eye View IPM Radar (Bottom Right)
        if homography_view:
            try:
                radar_inset = calibrator.render_bird_eye_inset(tracked_objects, inset_w=170, inset_h=240)
                rh, rw = radar_inset.shape[:2]
                rx_pos = frame_w - rw - 10
                ry_pos = frame_h - rh - 10
                display_frame[ry_pos:ry_pos + rh, rx_pos:rx_pos + rw] = radar_inset
            except Exception:
                pass

        if out_writer is not None:
            out_writer.write(display_frame)

        if args.show:
            cv2.imshow("AI Speed Detection System V2", display_frame)
            key = cv2.waitKey(0 if paused else max(1, delay_ms)) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n User requested exit (Q pressed)...")
                break
            elif key == ord(' '):
                paused = not paused
                print(" ⏸️ Paused" if paused else " ▶️ Resumed")
            elif key == ord('n') or key == ord('N'):
                night_mode_active = not night_mode_active
                print(f" 🌙 Night Vision / CLAHE: {'FORCED ON' if night_mode_active else 'AUTO'}")
            elif key == ord('h') or key == ord('H'):
                homography_view = not homography_view
                print(f" 📐 Homography / IPM Mode: {'ON' if homography_view else 'OFF'}")
            elif key == ord('t') or key == ord('T'):
                new_type = "centroid" if tracker_type == "bytetrack" else "bytetrack"
                tracker_type = new_type
                tracker = get_tracker(tracker_type=new_type, max_disappeared=MAX_DISAPPEARED_FRAMES)
                print(f" 🔄 Switched Tracker to: {tracker_type.upper()}")
            elif key == ord('o') or key == ord('O'):
                orientation = "vertical" if orientation == "horizontal" else "horizontal"
                print(f" 🔄 Switched orientation: {orientation.upper()}")
            elif key == ord('s') or key == ord('S'):
                snap_file = os.path.join("output", f"screenshot_frame_{frame_idx}.jpg")
                cv2.imwrite(snap_file, display_frame)
                print(f" 📸 Screenshot saved: {snap_file}")
            elif key == ord('1'):
                line1_ratio = max(0.05, line1_ratio - 0.02)
            elif key == ord('2'):
                line1_ratio = min(line2_ratio - 0.05, line1_ratio + 0.02)
            elif key == ord('3'):
                line2_ratio = max(line1_ratio + 0.05, line2_ratio - 0.02)
            elif key == ord('4'):
                line2_ratio = min(0.95, line2_ratio + 0.02)

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    # --- STEP 7: EXPORT LOGS & ANALYTICS ---
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_path = os.path.join("logs", f"speed_log_{today_str}.csv")
    db.export_csv(csv_path)

    json_path = os.path.join("logs", "session_summary.json")
    summary_stats = {
        "video": video_path,
        "processed_frames": frame_idx,
        "total_vehicles_tracked": tracker.next_object_id,
        "speeds_measured": speeds_count,
        "average_speed_kmh": round(avg_speed, 1),
        "maximum_speed_kmh": round(max_speed, 1),
        "speed_violations_count": over_limit_count,
        "tracker_used": tracker_type,
        "homography_used": homography_view,
        "night_mode_used": night_mode_active
    }
    db.export_json_summary(json_path, summary_stats)

    # Multi-panel Matplotlib Analytics
    analytics_path = os.path.join("output", "session_analytics.png")
    generate_session_analytics(captured_speeds, class_counts, overspeed_saved, args.speed_limit, analytics_path)

    # Legacy histogram path for backward compatibility
    histogram_path = os.path.join("output", "speed_histogram.png")
    if os.path.exists(analytics_path) and not os.path.exists(histogram_path):
        try:
            import shutil
            shutil.copy(analytics_path, histogram_path)
        except Exception:
            pass

    min_speed = float(np.min(captured_speeds)) if speeds_count > 0 else 0.0

    print("\n" + "═" * 52)
    print(" ╔══════════════════════════════════════════════════╗")
    print(" ║       SESSION V2 MULTI-TECH SUMMARY REPORT       ║")
    print(" ╠══════════════════════════════════════════════════╣")
    print(f" ║  Tracker Core:          {tracker_type.upper():<24} ║")
    print(f" ║  Total vehicles tracked:{tracker.next_object_id:<24} ║")
    print(f" ║  Speeds captured:       {speeds_count:<24} ║")
    print(f" ║  Average speed:         {avg_speed:<8.1f} km/h             ║")
    print(f" ║  Maximum speed:         {max_speed:<8.1f} km/h             ║")
    print(f" ║  Minimum speed:         {min_speed:<8.1f} km/h             ║")
    print(f" ║  Over speed limit:      {over_limit_count:<24} ║")
    print(f" ║  E-Challan Citations:   {len(overspeed_saved):<24} ║")
    print(f" ║  SQLite Database:       {'logs/violations.db':<24} ║")
    print(f" ║  Analytics Report:      {'output/session_analytics.png':<24} ║")
    print(f" ║  Video Rendered:        {output_path:<24} ║")
    print(" ╚══════════════════════════════════════════════════╝")
    print("═" * 52 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Vehicle Speed Detection System V2 (6 Next-Gen Technologies)")
    parser.add_argument("--video", required=True, help="Path to input traffic video file")
    parser.add_argument("--output", default=None, help="Path to save output video (optional)")
    parser.add_argument("--tracker", choices=["bytetrack", "centroid"], default="bytetrack", help="Tracking engine: 'bytetrack' (Kalman+IoU) or 'centroid' (default: bytetrack)")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal", help="Virtual line orientation (default: horizontal)")
    parser.add_argument("--homography", action="store_true", default=False, help="Enable 4-point IPM homography for true ground speed")
    parser.add_argument("--night-mode", action="store_true", default=False, help="Force low-light CLAHE enhancement & glare suppression")
    parser.add_argument("--alpr", action="store_true", default=True, help="Enable Automatic License Plate Recognition & E-Challan cards (default: True)")
    parser.add_argument("--no-alpr", action="store_false", dest="alpr", help="Disable ALPR module")
    parser.add_argument("--db", default="logs/violations.db", help="Path to SQLite audit database (default: logs/violations.db)")
    parser.add_argument("--distance", type=float, default=KNOWN_DISTANCE_METERS, help=f"Distance between reference lines in meters (default: {KNOWN_DISTANCE_METERS})")
    parser.add_argument("--speed-limit", type=float, default=SPEED_LIMIT_KMH, help=f"Speed limit in km/h (default: {SPEED_LIMIT_KMH})")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help=f"YOLO detection confidence (default: {CONFIDENCE_THRESHOLD})")
    parser.add_argument("--show", action="store_true", default=True, help="Show live preview window (default: True)")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Run headlessly without GUI preview")

    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

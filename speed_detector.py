# ============================================================
# Multi-Vehicle Speed Detection System
# ============================================================
# Team: Project Exhibition Team
# Project Exhibition 2026
#
# Installation:
#   python3 -m pip install ultralytics opencv-python numpy matplotlib
#
# Usage:
#   python3 speed_detector.py --video videos/sample_traffic.mp4
#   python3 speed_detector.py --video videos/sample_traffic.mp4 --orientation vertical
#
# How it works:
#   1. YOLOv8n detects vehicles in every frame
#   2. A simple centroid tracker assigns persistent IDs
#   3. Two virtual lines (Horizontal or Vertical) measure crossing times
#   4. Speed = known distance / crossing time (m/s converted to km/h)
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

# Import custom centroid tracker
from tracker import CentroidTracker

# Try importing ultralytics YOLO
try:
    from ultralytics import YOLO
except ImportError:
    print("Error: 'ultralytics' module not found. Please install it using: python3 -m pip install ultralytics")
    sys.exit(1)

# Try importing matplotlib for histogram generation
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# ===== CONFIGURATION =====
MODEL_NAME = "yolov8n.pt"            # YOLOv8 nano model (fast & lightweight)
CONFIDENCE_THRESHOLD = 0.35          # Minimum detection confidence threshold
VEHICLE_CLASSES = [2, 3, 5, 7]        # COCO IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
CLASS_NAMES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Color palette for vehicle classes (BGR)
CLASS_COLORS = {
    "Car": (255, 191, 0),        # Deep Sky Blue / Gold
    "Motorcycle": (255, 0, 255), # Magenta
    "Bus": (0, 255, 255),        # Yellow
    "Truck": (0, 165, 255)       # Orange
}

# Default Virtual Line positions (as fraction of frame dimensions 0.0 - 1.0)
LINE_1_POSITION = 0.35               # Entry line default (35% mark)
LINE_2_POSITION = 0.65               # Exit line default (65% mark)
DEFAULT_ORIENTATION = "horizontal"   # Default orientation ("horizontal" or "vertical")

# Real-world distance between the two reference lines in METERS
KNOWN_DISTANCE_METERS = 15.0

# Tracker parameters
MAX_DISAPPEARED_FRAMES = 40          # Frames before a lost vehicle is removed
MAX_MATCH_DISTANCE = 80              # Max pixel distance for centroid matching

# Speed rules
SPEED_LIMIT_KMH = 60.0               # Speed limit threshold in km/h
MIN_REASONABLE_SPEED = 5.0           # Ignore lower values (stationary/noise)
MAX_REASONABLE_SPEED = 200.0         # Ignore higher values (tracking glitch)

# Display Colors (BGR format)
LINE_COLOR_1 = (0, 255, 255)         # Yellow for Entry Line
LINE_COLOR_2 = (0, 0, 255)           # Red for Exit Line
SAFE_SPEED_COLOR = (0, 255, 0)       # Green for normal speed
OVER_SPEED_COLOR = (0, 0, 255)       # Red for overspeeding
TEXT_COLOR = (255, 255, 255)         # White text
PANEL_BG_COLOR = (20, 20, 20)        # Dark gray background for HUD


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


def draw_label(frame, text, x, y, bg_color, text_color=(255, 255, 255), font_scale=0.55, thickness=1):
    """Draw text with a solid filled background box for high readability."""
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(frame, (x, y - text_h - baseline - 4), (x + text_w + 6, y + 2), bg_color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - baseline - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)


def detect_vehicles(model, frame, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Run YOLOv8 object detection on a single frame and extract vehicle bounding boxes.
    """
    results = model(frame, conf=conf_threshold, verbose=False)[0]
    detections = []
    det_classes = []

    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append((x1, y1, x2, y2))
            det_classes.append(CLASS_NAMES.get(cls_id, "Vehicle"))

    return detections, det_classes


def draw_dashboard(frame, stats):
    """
    Draw a semi-transparent HUD dashboard panel in the top-left corner.
    """
    panel_w = 340
    panel_h = 240
    overlay = frame.copy()

    # Draw dark panel rectangle
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), PANEL_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (100, 100, 100), 1)

    # Title
    cv2.putText(frame, "SPEED DETECTION SYSTEM", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (20, 42), (10 + panel_w - 10, 42), (100, 100, 100), 1)

    lines = [
        f"Line Mode: {stats['orientation'].upper()}",
        f"Vehicles Tracked: {stats['tracked_total']}",
        f"Currently in Zone: {stats['in_zone']}",
        f"Speeds Measured: {stats['speeds_count']}",
        f"Avg Speed: {stats['avg_speed']:.1f} km/h" if stats['speeds_count'] > 0 else "Avg Speed: -- km/h",
        f"Max Speed: {stats['max_speed']:.1f} km/h" if stats['speeds_count'] > 0 else "Max Speed: -- km/h",
        f"Over Speed Limit: {stats['over_limit_count']}",
        f"Processing FPS: {stats['fps']:.1f}"
    ]

    y = 65
    for i, line in enumerate(lines):
        color = TEXT_COLOR
        if "Over Speed Limit" in line and stats['over_limit_count'] > 0:
            color = (0, 0, 255)
        elif "Line Mode" in line:
            color = (0, 255, 255)
        elif "Processing FPS" in line:
            color = (180, 180, 180)

        cv2.putText(frame, line, (22, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 20


def draw_vehicle_count_bar(frame, counts):
    """
    Draw a vehicle class count summary on the top-right side.
    """
    h, w = frame.shape[:2]
    panel_w = 180
    panel_h = 110
    x_start = w - panel_w - 10
    y_start = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x_start, y_start), (x_start + panel_w, y_start + panel_h), PANEL_BG_COLOR, cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (x_start, y_start), (x_start + panel_w, y_start + panel_h), (100, 100, 100), 1)

    cv2.putText(frame, "VEHICLE COUNTS", (x_start + 12, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    y = y_start + 48
    for cls_name in ["Car", "Truck", "Bus", "Motorcycle"]:
        count = counts.get(cls_name, 0)
        color = CLASS_COLORS.get(cls_name, (255, 255, 255))
        cv2.rectangle(frame, (x_start + 12, y - 10), (x_start + 22, y), color, cv2.FILLED)
        cv2.putText(frame, f"{cls_name}s: {count}", (x_start + 30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        y += 18


def save_speed_log(log_filepath, timestamp_str, vehicle_id, vehicle_class, speed_kmh, speed_limit):
    """Append a speed measurement to the CSV log file."""
    over_limit = "Yes" if speed_kmh > speed_limit else "No"
    file_exists = os.path.isfile(log_filepath)

    with open(log_filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Vehicle_ID", "Vehicle_Type", "Speed_KMH", "Over_Limit"])
        writer.writerow([timestamp_str, vehicle_id, vehicle_class, f"{speed_kmh:.1f}", over_limit])


def generate_speed_histogram(speeds, output_path):
    """Generate and save a matplotlib speed distribution histogram."""
    if not HAS_MATPLOTLIB or not speeds:
        return

    try:
        plt.figure(figsize=(8, 5))
        plt.hist(speeds, bins=12, color='skyblue', edgecolor='black', alpha=0.8)
        plt.axvline(SPEED_LIMIT_KMH, color='red', linestyle='dashed', linewidth=2, label=f'Speed Limit ({SPEED_LIMIT_KMH} km/h)')
        plt.title('Vehicle Speed Distribution Summary', fontsize=14, fontweight='bold')
        plt.xlabel('Speed (km/h)', fontsize=12)
        plt.ylabel('Vehicle Count', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"📊 Speed histogram saved to: {output_path}")
    except Exception as e:
        print(f"Warning: Failed to generate speed histogram: {e}")


def process_video(args):
    """Main video processing function."""
    video_path = args.video
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file '{video_path}' not found. Please check the file path.")
        return

    # Ensure output directories exist
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/overspeeding", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    print("\n" + "=" * 60)
    print(" 🚗 MULTI-VEHICLE SPEED DETECTION SYSTEM")
    print("=" * 60)
    print(f" Loading YOLOv8 model '{MODEL_NAME}'...")

    try:
        model = YOLO(MODEL_NAME)
        print(" Model loaded successfully! ✅")
    except Exception as e:
        print(f"❌ Error downloading/loading YOLO model: {e}")
        print(" Please check your internet connection or download 'yolov8n.pt' manually.")
        return

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: OpenCV cannot open video file '{video_path}'.")
        return

    # Extract video properties
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or math.isnan(fps):
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    orientation = args.orientation.lower()

    print(f" Video Source: {video_path}")
    print(f" Resolution: {frame_w}x{frame_h} | FPS: {fps:.2f} | Duration: {duration_sec:.1f}s ({total_frames} frames)")
    print(f" Line Mode: {orientation.upper()} | Reference Distance: {args.distance:.1f} meters | Speed Limit: {args.speed_limit:.1f} km/h")
    print("-" * 60)
    print(" Controls: [SPACE] Pause | [Q] Quit | [O] Toggle Lines (Horiz/Vert) | [1/2/3/4] Move Lines")
    print("-" * 60 + "\n")

    # Set up video writer if output requested
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

    # Initialize CSV log file path
    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filepath = os.path.join("logs", f"speed_log_{today_str}.csv")

    # Tracker & state tracking structures
    tracker = CentroidTracker(max_disappeared=MAX_DISAPPEARED_FRAMES, max_distance=MAX_MATCH_DISTANCE)

    line1_ratio = LINE_1_POSITION
    line2_ratio = LINE_2_POSITION

    line1_cross_time = {}
    line2_cross_time = {}
    vehicle_speeds = {}
    vehicle_classes = {}
    crossed_line1 = set()
    crossed_line2 = set()
    overspeed_saved = set()
    vehicle_trails = defaultdict(lambda: deque(maxlen=30))
    class_counts = defaultdict(int)

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

        current_video_time = frame_idx / fps

        # 1. Detect vehicles using YOLO
        raw_detections, raw_det_classes = detect_vehicles(model, frame, conf_threshold=args.conf)

        # 2. Update tracker
        tracked_objects = tracker.update(raw_detections)

        # Class matching
        for rect, cls_name in zip(raw_detections, raw_det_classes):
            rcx = int((rect[0] + rect[2]) / 2.0)
            rcy = int((rect[1] + rect[3]) / 2.0)
            for vid, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
                if math.hypot(cx - rcx, cy - rcy) < 25:
                    if vid not in vehicle_classes:
                        vehicle_classes[vid] = cls_name
                        class_counts[cls_name] += 1

        # Calculate Line pixel positions based on orientation
        if orientation == "horizontal":
            line1_pos = int(frame_h * line1_ratio)
            line2_pos = int(frame_h * line2_ratio)
        else:
            line1_pos = int(frame_w * line1_ratio)
            line2_pos = int(frame_w * line2_ratio)

        entry_line = min(line1_pos, line2_pos)
        exit_line = max(line1_pos, line2_pos)

        # 3. Check line crossings & calculate speed
        in_zone_count = 0
        for vehicle_id, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
            # Get previous centroid coordinate
            has_history = len(vehicle_trails[vehicle_id]) > 0
            prev_pt = vehicle_trails[vehicle_id][-1] if has_history else (cx, cy)
            vehicle_trails[vehicle_id].append((cx, cy))
            
            curr_coord = cy if orientation == "horizontal" else cx
            prev_coord = prev_pt[1] if orientation == "horizontal" else prev_pt[0]

            # Count vehicles currently between Line 1 and Line 2
            if entry_line <= curr_coord <= exit_line:
                in_zone_count += 1

            # Check boundary crossings (only if we have previous frame history)
            if has_history:
                # Forward motion (increasing coordinate: top->bottom or left->right)
                if prev_coord <= entry_line < curr_coord and vehicle_id not in line1_cross_time:
                    line1_cross_time[vehicle_id] = current_video_time
                    crossed_line1.add(vehicle_id)
                elif prev_coord <= exit_line < curr_coord and vehicle_id not in line2_cross_time:
                    line2_cross_time[vehicle_id] = current_video_time
                    crossed_line2.add(vehicle_id)

                # Reverse motion (decreasing coordinate: bottom->top or right->left)
                elif curr_coord <= exit_line < prev_coord and vehicle_id not in line2_cross_time:
                    line2_cross_time[vehicle_id] = current_video_time
                    crossed_line2.add(vehicle_id)
                elif curr_coord <= entry_line < prev_coord and vehicle_id not in line1_cross_time:
                    line1_cross_time[vehicle_id] = current_video_time
                    crossed_line1.add(vehicle_id)

            # Calculate Speed if vehicle crossed BOTH lines
            if vehicle_id in line1_cross_time and vehicle_id in line2_cross_time and vehicle_id not in vehicle_speeds:
                dt = abs(line2_cross_time[vehicle_id] - line1_cross_time[vehicle_id])
                if dt > 0.05:
                    speed_ms = args.distance / dt
                    speed_kmh = speed_ms * 3.6

                    if MIN_REASONABLE_SPEED <= speed_kmh <= MAX_REASONABLE_SPEED:
                        vehicle_speeds[vehicle_id] = round(speed_kmh, 1)
                        v_class = vehicle_classes.get(vehicle_id, "Vehicle")
                        time_stamp_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        # Log to CSV
                        save_speed_log(log_filepath, time_stamp_now, vehicle_id, v_class, speed_kmh, args.speed_limit)

                        # Print real-time event
                        over_tag = " [OVER SPEED! ⚠️]" if speed_kmh > args.speed_limit else ""
                        print(f"⚡ [EVENT] Vehicle ID #{vehicle_id} ({v_class}) | Speed: {speed_kmh:.1f} km/h{over_tag}")

                        # Save snapshot if overspeeding
                        if speed_kmh > args.speed_limit and vehicle_id not in overspeed_saved:
                            overspeed_saved.add(vehicle_id)
                            try:
                                crop_x1 = max(0, x1 - 10)
                                crop_y1 = max(0, y1 - 10)
                                crop_x2 = min(frame_w, x2 + 10)
                                crop_y2 = min(frame_h, y2 + 10)
                                crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                                if crop_img.size > 0:
                                    snap_name = f"overspeed_id{vehicle_id}_{int(speed_kmh)}kmh.jpg"
                                    cv2.imwrite(os.path.join("output", "overspeeding", snap_name), crop_img)
                            except Exception as e:
                                pass

        # Calculate FPS
        now = time.time()
        if now - last_fps_time >= 1.0:
            current_fps = fps_counter / (now - last_fps_time)
            fps_counter = 0
            last_fps_time = now

        # 4. DRAW OVERLAYS ON FRAME
        display_frame = frame.copy()

        # Draw entry and exit lines
        if orientation == "horizontal":
            draw_dashed_line(display_frame, (0, line1_pos), (frame_w, line1_pos), LINE_COLOR_1, thickness=2)
            draw_label(display_frame, f"LINE 1 (ENTRY) - {int(line1_ratio * 100)}%", 15, line1_pos - 5, LINE_COLOR_1, (0, 0, 0))

            draw_dashed_line(display_frame, (0, line2_pos), (frame_w, line2_pos), LINE_COLOR_2, thickness=2)
            draw_label(display_frame, f"LINE 2 (EXIT) - {int(line2_ratio * 100)}%", 15, line2_pos - 5, LINE_COLOR_2, (255, 255, 255))
        else:
            draw_dashed_line(display_frame, (line1_pos, 0), (line1_pos, frame_h), LINE_COLOR_1, thickness=2)
            draw_label(display_frame, f"LINE 1 (ENTRY) - {int(line1_ratio * 100)}%", line1_pos + 5, 260, LINE_COLOR_1, (0, 0, 0))
            draw_dashed_line(display_frame, (line2_pos, 0), (line2_pos, frame_h), LINE_COLOR_2, thickness=2)
            draw_label(display_frame, f"LINE 2 (EXIT) - {int(line2_ratio * 100)}%", line2_pos + 5, 300, LINE_COLOR_2, (255, 255, 255))

        # Draw motion trails & tracked vehicle boxes
        for vehicle_id, (cx, cy, x1, y1, x2, y2) in tracked_objects.items():
            v_class = vehicle_classes.get(vehicle_id, "Vehicle")

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
                box_color = SAFE_SPEED_COLOR
                speed_str = ""

            # Motion trail tail
            trail_pts = list(vehicle_trails[vehicle_id])
            for i in range(1, len(trail_pts)):
                pt_a = trail_pts[i - 1]
                pt_b = trail_pts[i]
                thickness = int(np.sqrt(30 / float(i + 1)) * 1.5)
                cv2.line(display_frame, pt_a, pt_b, box_color, max(1, thickness))

            # Bounding box & center circle
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(display_frame, (cx, cy), 4, box_color, -1)

            # Label text above box
            label_text = f"ID:{vehicle_id} {v_class}"
            if speed_str:
                label_text += f" | {speed_str}"

            draw_label(display_frame, label_text, x1, y1 - 5, box_color, (255, 255, 255) if box_color != LINE_COLOR_1 else (0, 0, 0))

        # Statistics HUD Dashboard
        captured_speeds = list(vehicle_speeds.values())
        speeds_count = len(captured_speeds)
        avg_speed = float(np.mean(captured_speeds)) if speeds_count > 0 else 0.0
        max_speed = float(np.max(captured_speeds)) if speeds_count > 0 else 0.0
        over_limit_count = sum(1 for s in captured_speeds if s > args.speed_limit)

        dashboard_stats = {
            "orientation": orientation,
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

        if out_writer is not None:
            out_writer.write(display_frame)

        if args.show:
            cv2.imshow("Multi-Vehicle Speed Detection System", display_frame)
            key = cv2.waitKey(0 if paused else max(1, delay_ms)) & 0xFF

            if key == ord('q') or key == ord('Q') or key == 27:
                print("\n Exiting user request (Q pressed)...")
                break
            elif key == ord(' '):
                paused = not paused
                print(" ⏸️ Paused" if paused else " ▶️ Resumed")
            elif key == ord('o') or key == ord('O'):
                orientation = "vertical" if orientation == "horizontal" else "horizontal"
                print(f" 🔄 Switched line orientation to: {orientation.upper()}")
            elif key == ord('s') or key == ord('S'):
                snap_file = os.path.join("output", f"screenshot_frame_{frame_idx}.jpg")
                cv2.imwrite(snap_file, display_frame)
                print(f" 📸 Screenshot saved to: {snap_file}")
            elif key == ord('+') or key == ord('='):
                delay_ms = max(1, delay_ms - 5)
            elif key == ord('-') or key == ord('_'):
                delay_ms = min(200, delay_ms + 5)
            elif key == ord('1'):
                line1_ratio = max(0.05, line1_ratio - 0.02)
                print(f" Adjusted Line 1 Position: {line1_ratio:.2f}")
            elif key == ord('2'):
                line1_ratio = min(line2_ratio - 0.05, line1_ratio + 0.02)
                print(f" Adjusted Line 1 Position: {line1_ratio:.2f}")
            elif key == ord('3'):
                line2_ratio = max(line1_ratio + 0.05, line2_ratio - 0.02)
                print(f" Adjusted Line 2 Position: {line2_ratio:.2f}")
            elif key == ord('4'):
                line2_ratio = min(0.95, line2_ratio + 0.02)
                print(f" Adjusted Line 2 Position: {line2_ratio:.2f}")

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    histogram_path = os.path.join("output", "speed_histogram.png")
    generate_speed_histogram(captured_speeds, histogram_path)

    min_speed = float(np.min(captured_speeds)) if speeds_count > 0 else 0.0

    print("\n" + "═" * 44)
    print(" ╔══════════════════════════════════════════╗")
    print(" ║         SESSION SUMMARY REPORT           ║")
    print(" ╠══════════════════════════════════════════╣")
    print(f" ║  Total vehicles tracked:  {tracker.next_object_id:<14} ║")
    print(f" ║  Speeds captured:         {speeds_count:<14} ║")
    print(f" ║  Average speed:           {avg_speed:<8.1f} km/h   ║")
    print(f" ║  Maximum speed:           {max_speed:<8.1f} km/h   ║")
    print(f" ║  Minimum speed:           {min_speed:<8.1f} km/h   ║")
    print(f" ║  Over speed limit:        {over_limit_count:<14} ║")
    print(f" ║  Output video saved:      {output_path:<14} ║")
    print(f" ║  Speed log saved:         {log_filepath:<14} ║")
    print(" ╚══════════════════════════════════════════╝")
    print("═" * 44 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Vehicle Speed Detection System (YOLOv8 + Centroid Tracker)")
    parser.add_argument("--video", required=True, help="Path to input traffic video file")
    parser.add_argument("--output", default=None, help="Path to save output video (optional)")
    parser.add_argument("--orientation", choices=["horizontal", "vertical"], default="horizontal", help="Reference line orientation: 'horizontal' or 'vertical' (default: horizontal)")
    parser.add_argument("--show", action="store_true", default=True, help="Show live preview window (default: True)")
    parser.add_argument("--no-show", action="store_false", dest="show", help="Disable live preview window")
    parser.add_argument("--distance", type=float, default=KNOWN_DISTANCE_METERS, help=f"Real distance between lines in meters (default: {KNOWN_DISTANCE_METERS})")
    parser.add_argument("--speed-limit", type=float, default=SPEED_LIMIT_KMH, help=f"Speed limit in km/h (default: {SPEED_LIMIT_KMH})")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD, help=f"YOLO confidence threshold (default: {CONFIDENCE_THRESHOLD})")

    args = parser.parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

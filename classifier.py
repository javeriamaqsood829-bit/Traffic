# ============================================================
# Multi-Vehicle Speed Detection System — Classification Module
# ============================================================
# Fine-Grained Vehicle Classification & Regional Vehicle Support
# (Auto-Rickshaws, Three-Wheelers, LCVs, Buses, Trucks, Bikes, Cars)
# + Multi-Lane Spatial Assignment.
# ============================================================

import cv2
import numpy as np


class VehicleClassifier:
    """
    Enhanced vehicle classification engine.
    Refines YOLOv8 standard COCO classes with regional vehicle heuristics
    (such as Indian Auto-Rickshaws and Light Commercial Vehicles) and assigns lanes.
    """

    # Extended vehicle class palette (BGR)
    CLASS_COLORS = {
        "Car": (255, 191, 0),             # Deep Sky Blue / Gold
        "Motorcycle": (255, 0, 255),      # Magenta
        "Bus": (0, 255, 255),             # Yellow
        "Truck": (0, 165, 255),           # Orange
        "Auto-Rickshaw": (0, 230, 115),   # Bright Mint / Green (Classic Three-Wheeler)
        "Commercial": (200, 100, 255),    # Light Purple / Violet
        "Vehicle": (220, 220, 220)        # Neutral Gray
    }

    def __init__(self, num_lanes=3):
        self.num_lanes = num_lanes

    def is_auto_rickshaw(self, vehicle_crop, base_class, confidence):
        """
        Heuristic detector for Auto-Rickshaws / Three-Wheelers.
        Examines bounding box aspect ratio and characteristic yellow-green color ratio.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return False

        vh, vw = vehicle_crop.shape[:2]
        if vh < 25 or vw < 20:
            return False

        aspect_ratio = vw / float(vh)

        # Three-wheelers have a distinct tall, compact profile (aspect ratio 0.65 - 0.98)
        # Unlike wide low cars (aspect ratio 1.2 - 2.0)
        is_squarish_profile = 0.62 <= aspect_ratio <= 1.05

        if not is_squarish_profile:
            return False

        # If YOLO detected it as motorcycle or car with borderline confidence
        if base_class in ["Car", "Motorcycle"] and confidence < 0.75:
            # Check for yellow or green color dominance (classic autorickshaw paint)
            hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
            # Yellow hue: [20, 100, 100] to [32, 255, 255]
            yellow_mask = cv2.inRange(hsv, np.array([18, 60, 60]), np.array([36, 255, 255]))
            # Green hue: [35, 50, 50] to [85, 255, 255]
            green_mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))

            total_pixels = float(vh * vw)
            color_ratio = (np.count_nonzero(yellow_mask) + np.count_nonzero(green_mask)) / total_pixels

            if color_ratio > 0.08:
                return True

        return False

    def classify(self, vehicle_crop, base_class_id, confidence, bbox, frame_w):
        """
        Enrich and refine vehicle detection with sub-category and lane metadata.
        
        Args:
            vehicle_crop (ndarray): Cropped vehicle image
            base_class_id (int): YOLO class ID (2, 3, 5, 7)
            confidence (float): Detection confidence score
            bbox (tuple): (x1, y1, x2, y2)
            frame_w (int): Full frame width for lane calculation
            
        Returns:
            dict: {
                "class_name": str,
                "confidence": float,
                "color": tuple,
                "lane_id": str,
                "lane_num": int
            }
        """
        coco_map = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
        base_name = coco_map.get(base_class_id, "Car")

        # Refine for regional vehicle classes
        if self.is_auto_rickshaw(vehicle_crop, base_name, confidence):
            final_class = "Auto-Rickshaw"
        elif base_name == "Truck":
            # Distinguish light commercial vehicle / van from heavy truck
            x1, y1, x2, y2 = bbox
            box_area = (x2 - x1) * (y2 - y1)
            final_class = "Commercial" if box_area < 25000 else "Truck"
        else:
            final_class = base_name

        # Calculate spatial lane assignment across road width
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        lane_width = frame_w / float(self.num_lanes)
        lane_index = min(self.num_lanes - 1, max(0, int(cx // lane_width)))
        lane_labels = ["Lane 1 (Left)", "Lane 2 (Center)", "Lane 3 (Right)"]
        lane_label = lane_labels[lane_index] if lane_index < len(lane_labels) else f"Lane {lane_index + 1}"

        return {
            "class_name": final_class,
            "confidence": float(confidence),
            "color": self.CLASS_COLORS.get(final_class, (255, 255, 255)),
            "lane_id": lane_label,
            "lane_num": lane_index + 1
        }


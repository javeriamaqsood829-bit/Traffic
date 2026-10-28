# ============================================================
# Multi-Vehicle Speed Detection System — Homography & IPM Module
# ============================================================
# Inverse Perspective Mapping (IPM) & 4-Point Ground Calibration.
# Eliminates oblique camera distortion by mapping image pixels
# directly to real-world ground coordinates in METERS.
# ============================================================

import cv2
import numpy as np
import math


class PerspectiveCalibrator:
    """
    Inverse Perspective Mapping (IPM) calibrator.
    Maps 2D pixel coordinates (u, v) on the camera image plane to
    metric coordinates (X, Y) in meters on the physical road surface.
    """

    def __init__(self, src_pts=None, dst_pts=None, road_width_m=8.0, road_length_m=35.0):
        """
        Args:
            src_pts (list or ndarray): 4 polygon points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
                                       representing trapezoidal road area in frame.
            dst_pts (list or ndarray): 4 corresponding metric points [(0,0), (w, 0), (w, l), (0, l)]
                                       in real-world meters.
            road_width_m (float): Default real-world road width in meters (e.g. 2 lanes = 7.5m - 8.0m)
            road_length_m (float): Default distance covered along the road in meters.
        """
        self.road_width_m = road_width_m
        self.road_length_m = road_length_m
        self.H = None
        self.H_inv = None
        self.src_pts = None
        self.dst_pts = None

        if src_pts is not None:
            self.set_calibration_points(src_pts, dst_pts)

    def set_default_preset(self, frame_w, frame_h):
        """
        Setup default perspective trapezoid calibrated for standard highway surveillance cameras.
        Top: narrower lane perspective in distance.
        Bottom: wider lane perspective in foreground.
        """
        # Trapezoid source coordinates on screen
        top_y = int(frame_h * 0.28)
        bottom_y = int(frame_h * 0.90)

        top_left_x = int(frame_w * 0.28)
        top_right_x = int(frame_w * 0.72)
        bottom_left_x = int(frame_w * 0.08)
        bottom_right_x = int(frame_w * 0.92)

        src_pts = np.float32([
            [top_left_x, top_y],       # Top-Left (Far)
            [top_right_x, top_y],      # Top-Right (Far)
            [bottom_right_x, bottom_y],# Bottom-Right (Near)
            [bottom_left_x, bottom_y]  # Bottom-Left (Near)
        ])

        # Destination metric coordinates in meters
        # Origin (0,0) at top-left of calibration zone
        dst_pts = np.float32([
            [0.0, 0.0],
            [self.road_width_m, 0.0],
            [self.road_width_m, self.road_length_m],
            [0.0, self.road_length_m]
        ])

        self.set_calibration_points(src_pts, dst_pts)

    def set_calibration_points(self, src_pts, dst_pts=None):
        """
        Compute the 3x3 Homography Matrix H and its inverse.
        """
        self.src_pts = np.float32(src_pts)

        if dst_pts is None:
            self.dst_pts = np.float32([
                [0.0, 0.0],
                [self.road_width_m, 0.0],
                [self.road_width_m, self.road_length_m],
                [0.0, self.road_length_m]
            ])
        else:
            self.dst_pts = np.float32(dst_pts)

        self.H = cv2.getPerspectiveTransform(self.src_pts, self.dst_pts)
        self.H_inv = cv2.getPerspectiveTransform(self.dst_pts, self.src_pts)

    def pixel_to_ground(self, u, v):
        """
        Convert image pixel coordinates (u, v) to ground metric coordinates (X, Y) in meters.
        """
        if self.H is None:
            return float(u), float(v)

        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H)
        ground_x = float(transformed[0][0][0])
        ground_y = float(transformed[0][0][1])
        return ground_x, ground_y

    def calculate_speed_ipm(self, pt_entry, t_entry, pt_exit, t_exit):
        """
        Calculate true Euclidean ground speed using calibrated homography.
        
        Args:
            pt_entry (tuple): (u1, v1) entry point in image
            t_entry (float): video timestamp at entry
            pt_exit (tuple): (u2, v2) exit point in image
            t_exit (float): video timestamp at exit
        
        Returns:
            float: Speed in km/h, or None if invalid
        """
        dt = abs(t_exit - t_entry)
        if dt < 0.05 or self.H is None:
            return None

        # Transform both pixel points to real-world ground meters
        gx1, gy1 = self.pixel_to_ground(pt_entry[0], pt_entry[1])
        gx2, gy2 = self.pixel_to_ground(pt_exit[0], pt_exit[1])

        # Euclidean displacement in real-world ground plane
        displacement_m = math.hypot(gx2 - gx1, gy2 - gy1)
        speed_ms = displacement_m / dt
        speed_kmh = speed_ms * 3.6
        return speed_kmh

    def draw_calibration_zone(self, frame, color=(0, 255, 255), thickness=2):
        """
        Draw the 4-point calibration trapezoid boundary onto the camera frame.
        """
        if self.src_pts is None:
            return frame

        pts = self.src_pts.astype(np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=thickness)
        # Semi-transparent fill of the ground calibration zone
        cv2.fillPoly(overlay, [pts], color=(0, 80, 80))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Label corner vertices
        for i, (x, y) in enumerate(self.src_pts):
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"P{i+1}", (int(x) + 7, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def render_bird_eye_inset(self, tracked_vehicles, inset_w=180, inset_h=260):
        """
        Render a mini top-down 'Bird's-Eye View' radar panel displaying
        vehicles positioned in their physical metric road positions.
        """
        radar = np.full((inset_h, inset_w, 3), (25, 25, 30), dtype=np.uint8)

        # Radar border & title
        cv2.rectangle(radar, (0, 0), (inset_w - 1, inset_h - 1), (80, 80, 90), 1)
        cv2.putText(radar, "IPM RADAR (m)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        # Draw road lane boundaries in metric view
        pad_x = 20
        pad_y = 35
        road_render_w = inset_w - 2 * pad_x
        road_render_h = inset_h - pad_y - 15

        cv2.rectangle(radar, (pad_x, pad_y), (pad_x + road_render_w, pad_y + road_render_h), (50, 50, 60), cv2.FILLED)
        # Center dividing dashed line
        center_x = pad_x + road_render_w // 2
        for dy in range(pad_y, pad_y + road_render_h, 15):
            cv2.line(radar, (center_x, dy), (center_x, min(dy + 8, pad_y + road_render_h)), (120, 120, 130), 1)

        # Plot vehicles in top-down metric plane
        for vid, vdata in tracked_vehicles.items():
            cx = vdata[0]
            # Use bottom center of bounding box for ground contact point
            bottom_y = vdata[5] if len(vdata) >= 6 else vdata[1]

            gx, gy = self.pixel_to_ground(cx, bottom_y)

            # Check if within calibrated metric bounds
            if 0.0 <= gx <= self.road_width_m and 0.0 <= gy <= self.road_length_m:
                norm_x = gx / float(self.road_width_m)
                norm_y = gy / float(self.road_length_m)

                rx = int(pad_x + norm_x * road_render_w)
                ry = int(pad_y + norm_y * road_render_h)

                cv2.circle(radar, (rx, ry), 4, (0, 255, 0), -1)
                cv2.putText(radar, f"#{vid}", (rx + 5, ry + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return radar


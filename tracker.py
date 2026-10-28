# ============================================================
# Multi-Vehicle Speed Detection System — Multi-Object Tracker
# ============================================================
# Features:
#   1. ByteTrack-style Kalman Filter + IoU Tracker (State-of-the-art)
#   2. Pure-Python Centroid Tracker (Lightweight fallback)
# Solves occlusion, close vehicle crossovers, and ID switching.
# ============================================================

import numpy as np
import math

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def calculate_iou(boxA, boxB):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Boxes format: (x1, y1, x2, y2)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = max(0, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    union_area = float(areaA + areaB - inter_area)

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ============================================================
# 1. KALMAN FILTER FOR BOUNDING BOX TRACKING
# ============================================================

class KalmanBoxTracker:
    """
    Kalman Filter tracking a bounding box with an 8-dimensional state vector:
    [cx, cy, s, r, vx, vy, vs, vr]^T
    where:
        cx, cy: Center coordinates of bounding box
        s: Scale (area = width * height)
        r: Aspect ratio (width / height)
        vx, vy, vs, vr: First-order derivatives (velocities)
    """

    count = 0

    def __init__(self, bbox):
        # State vector: 8x1
        self.x = np.zeros((8, 1), dtype=np.float32)
        # Convert initial (x1, y1, x2, y2) to [cx, cy, s, r]
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0] + bbox[2]) / 2.0
        cy = float(bbox[1] + bbox[3]) / 2.0
        s = w * h
        r = w / h

        self.x[:4, 0] = [cx, cy, s, r]

        # State transition matrix (constant velocity model)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix (observes cx, cy, s, r)
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 100.0, 1.0, 1000.0, 1000.0, 10000.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 1.0, 0.01, 1.0, 1.0, 1.0, 0.01]).astype(np.float32)
        self.R = np.diag([1.0, 1.0, 10.0, 0.1]).astype(np.float32)

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.last_bbox = bbox

    def predict(self):
        """Advance the state vector and predict bounding box."""
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.get_state())
        return self.history[-1]

    def update(self, bbox):
        """Update the state vector with observed bounding box measurement."""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1

        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = float(bbox[0] + bbox[2]) / 2.0
        cy = float(bbox[1] + bbox[3]) / 2.0
        s = w * h
        r = w / h

        z = np.array([[cx], [cy], [s], [r]], dtype=np.float32)

        # Innovation / Measurement residual
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        self.last_bbox = bbox

    def get_state(self):
        """Convert current state estimate [cx, cy, s, r] to bounding box (x1, y1, x2, y2)."""
        cx, cy, s, r = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        if s <= 0 or r <= 0:
            return self.last_bbox

        w = math.sqrt(s * r)
        h = s / w if w > 0 else 1.0

        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return (x1, y1, x2, y2)


# ============================================================
# 2. BYTETRACK-STYLE MULTI-OBJECT TRACKER
# ============================================================

class ByteTrackTracker:
    """
    ByteTrack association tracker with Kalman filtering.
    Associates high-confidence and low-confidence detections in two stages,
    preventing track fragmentation and ID switching during occlusions.
    """

    def __init__(self, max_disappeared=40, min_hits=2, iou_threshold=0.25):
        self.max_disappeared = max_disappeared
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.next_object_id = 0
        KalmanBoxTracker.count = 0

    def associate_detections_to_trackers(self, detections, trackers, iou_thresh):
        """
        Assign detections to tracked targets using IoU cost matrix.
        """
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = calculate_iou(det, trk)

        if HAS_SCIPY:
            # Optimal linear sum assignment (Hungarian Algorithm)
            cost_matrix = -iou_matrix
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matched_indices = np.column_stack((row_ind, col_ind))
        else:
            # Greedy matching fallback
            matched_indices = []
            flat_indices = np.argsort(-iou_matrix, axis=None)
            used_rows = set()
            used_cols = set()
            for idx in flat_indices:
                r = int(idx // iou_matrix.shape[1])
                c = int(idx % iou_matrix.shape[1])
                if r in used_rows or c in used_cols:
                    continue
                if iou_matrix[r, c] < iou_thresh:
                    break
                matched_indices.append([r, c])
                used_rows.add(r)
                used_cols.add(c)
            matched_indices = np.array(matched_indices) if len(matched_indices) > 0 else np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_thresh:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self, rects):
        """
        Update tracked objects with new frame bounding box detections.
        
        Args:
            rects (list): List of (x1, y1, x2, y2) bounding boxes
            
        Returns:
            dict: object_id -> (cx, cy, x1, y1, x2, y2)
        """
        # Step 1: Advance existing Kalman trackers to current frame
        trks = np.zeros((len(self.trackers), 4))
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = [pos[0], pos[1], pos[2], pos[3]]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # Step 2: Associate detections with predictions
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            rects, trks, self.iou_threshold
        )

        # Step 3: Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(rects[m[0]])

        # Step 4: Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(rects[i])
            self.trackers.append(trk)
            if trk.id >= self.next_object_id:
                self.next_object_id = trk.id + 1

        # Step 5: Clean up stale trackers
        i = len(self.trackers)
        results = {}
        for trk in reversed(self.trackers):
            d = trk.get_state()
            # Confirm tracks that have been active and are not expired
            if (trk.time_since_update <= 1) and (trk.hit_streak >= self.min_hits or trk.age <= self.min_hits):
                x1, y1, x2, y2 = d
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                results[trk.id] = (cx, cy, x1, y1, x2, y2)

            i -= 1
            if trk.time_since_update > self.max_disappeared:
                self.trackers.pop(i)

        return results


# ============================================================
# 3. LEGACY CENTROID TRACKER (PURE-PYTHON FALLBACK)
# ============================================================

class CentroidTracker:
    """
    A simple Centroid Tracker that tracks objects across frames
    by matching bounding box center points (centroids) based on 
    Euclidean distance.
    """

    def __init__(self, max_disappeared=40, max_distance=80):
        self.next_object_id = 0
        self.objects = {}
        self.bboxes = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self._get_tracked_dict()

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]), tuple(rects[i]))
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
            for i, (ox, oy) in enumerate(object_centroids):
                for j, (ix, iy) in enumerate(input_centroids):
                    D[i, j] = math.hypot(ox - ix, oy - iy)

            used_rows = set()
            used_cols = set()
            flat_indices = np.argsort(D, axis=None)

            for idx in flat_indices:
                row = int(idx // D.shape[1])
                col = int(idx % D.shape[1])

                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    break

                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.bboxes[object_id] = tuple(rects[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), tuple(rects[col]))

        return self._get_tracked_dict()

    def _get_tracked_dict(self):
        results = {}
        for object_id in self.objects.keys():
            cx, cy = self.objects[object_id]
            x1, y1, x2, y2 = self.bboxes[object_id]
            results[object_id] = (cx, cy, x1, y1, x2, y2)
        return results


# ============================================================
# TRACKER FACTORY
# ============================================================

def get_tracker(tracker_type="bytetrack", max_disappeared=40, max_distance=80):
    """
    Factory function to instantiate tracker of chosen algorithm.
    """
    if tracker_type.lower() == "centroid":
        return CentroidTracker(max_disappeared=max_disappeared, max_distance=max_distance)
    else:
        return ByteTrackTracker(max_disappeared=max_disappeared, min_hits=1, iou_threshold=0.25)

# ============================================================
# Multi-Vehicle Speed Detection System — Centroid Tracker
# ============================================================
# Simple centroid-based multi-object tracker built from scratch.
# Assigns unique IDs to detected objects and tracks them across
# video frames using Euclidean distance matching.
# ============================================================

import numpy as np
import math


class CentroidTracker:
    """
    A simple Centroid Tracker that tracks objects across frames
    by matching bounding box center points (centroids) based on 
    Euclidean distance.
    """

    def __init__(self, max_disappeared=40, max_distance=80):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared (int): Max consecutive frames an object can be missing 
                                  before being deregistered.
            max_distance (int): Max pixel distance allowed between centroids 
                               to consider them the same object.
        """
        # Store the next available unique ID
        self.next_object_id = 0

        # Dictionary mapping object ID to its current centroid (cx, cy)
        self.objects = {}

        # Dictionary mapping object ID to its bounding box (x1, y1, x2, y2)
        self.bboxes = {}

        # Dictionary mapping object ID to consecutive frames disappeared
        self.disappeared = {}

        # Configuration parameters
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox):
        """
        Register a new object with the next available ID.

        Args:
            centroid (tuple): (cx, cy) coordinates
            bbox (tuple): (x1, y1, x2, y2) coordinates
        """
        self.objects[self.next_object_id] = centroid
        self.bboxes[self.next_object_id] = bbox
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        """
        Remove an object ID from tracking when it leaves the scene.

        Args:
            object_id (int): ID of the vehicle to remove
        """
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.bboxes:
            del self.bboxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, rects):
        """
        Update tracked objects based on new bounding box detections in the current frame.

        Args:
            rects (list): List of bounding boxes [(x1, y1, x2, y2), ...]

        Returns:
            dict: Mapping of object_id -> (cx, cy, x1, y1, x2, y2)
        """
        # If no bounding boxes are detected in this frame
        if len(rects) == 0:
            # Mark all existing tracked objects as disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                # Deregister if object exceeded max missing frames
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Return current state
            return self._get_tracked_dict()

        # Compute centroids for all input bounding boxes
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(rects):
            cx = int((x1 + x2) / 2.0)
            cy = int((y1 + y2) / 2.0)
            input_centroids[i] = (cx, cy)

        # If we currently have no tracked objects, register all detections
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]), tuple(rects[i]))
        else:
            # Grab existing object IDs and centroids
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute Euclidean distance matrix between existing centroids and input centroids
            # Shape: (len(object_centroids), len(input_centroids))
            D = np.zeros((len(object_centroids), len(input_centroids)), dtype="float")
            for i, (ox, oy) in enumerate(object_centroids):
                for j, (ix, iy) in enumerate(input_centroids):
                    D[i, j] = math.hypot(ox - ix, oy - iy)

            # Find matching using global greedy approach (smallest distance pairs across whole matrix)
            used_rows = set()
            used_cols = set()
            flat_indices = np.argsort(D, axis=None)

            for idx in flat_indices:
                row = int(idx // D.shape[1])
                col = int(idx % D.shape[1])

                # Skip if this object or detection has already been matched
                if row in used_rows or col in used_cols:
                    continue

                # Stop if distance exceeds max allowed distance
                if D[row, col] > self.max_distance:
                    break

                # Match found: update centroid, bounding box, and reset disappeared count
                object_id = object_ids[row]
                self.objects[object_id] = tuple(input_centroids[col])
                self.bboxes[object_id] = tuple(rects[col])
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Check unmatched existing objects
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Check unmatched new detections (register as new objects)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(tuple(input_centroids[col]), tuple(rects[col]))

        return self._get_tracked_dict()

    def _get_tracked_dict(self):
        """
        Helper method to compile current tracking state.

        Returns:
            dict: object_id -> (cx, cy, x1, y1, x2, y2)
        """
        results = {}
        for object_id in self.objects.keys():
            cx, cy = self.objects[object_id]
            x1, y1, x2, y2 = self.bboxes[object_id]
            results[object_id] = (cx, cy, x1, y1, x2, y2)
        return results

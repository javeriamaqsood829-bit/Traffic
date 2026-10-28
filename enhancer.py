# ============================================================
# Multi-Vehicle Speed Detection System — Night Enhancer Module
# ============================================================
# Low-Light Adaptation, CLAHE Contrast Enhancement, and
# Headlight Glare Suppression for 24/7 CCTV Operation.
# ============================================================

import cv2
import numpy as np


class NightEnhancer:
    """
    Adaptive low-light and night vision enhancement engine.
    Enhances dark surveillance frames using CLAHE and suppresses
    headlight glare blooming to maintain tracking accuracy.
    """

    def __init__(self, clip_limit=2.5, tile_grid_size=(8, 8), glare_suppression=True):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.glare_suppression = glare_suppression
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def is_low_light(self, frame, threshold=75.0):
        """
        Check if the current frame is low-light / nighttime based on mean luminance.
        """
        if frame is None or frame.size == 0:
            return False

        # Convert to grayscale to evaluate mean brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))
        return mean_val < threshold

    def suppress_headlight_glare(self, frame):
        """
        Detect and suppress intense specular headlight bloom that blinds YOLO
        and causes false bounding box expansion.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # High brightness (V > 235) and low saturation (S < 60) represents white headlight glare
        glare_mask = cv2.inRange(hsv, np.array([0, 0, 235]), np.array([180, 65, 255]))

        # Smooth and dilate glare mask to capture bloom halo
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        glare_mask_dilated = cv2.dilate(glare_mask, kernel, iterations=1)
        glare_mask_blur = cv2.GaussianBlur(glare_mask_dilated, (15, 15), 0)

        # Blend compressed luminance into glare regions
        norm_mask = glare_mask_blur.astype(np.float32) / 255.0
        attenuated_v = cv2.multiply(v.astype(np.float32), 0.70).astype(np.uint8)

        v_suppressed = (v * (1.0 - norm_mask) + attenuated_v * norm_mask).astype(np.uint8)
        enhanced_hsv = cv2.merge([h, s, v_suppressed])
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    def enhance(self, frame, force_night=False, auto_detect=True):
        """
        Enhance video frame if low-light condition is detected or forced.
        
        Args:
            frame (ndarray): Input BGR frame
            force_night (bool): Explicitly apply night enhancement
            auto_detect (bool): Auto-detect whether frame is dark
            
        Returns:
            (ndarray, bool): (Enhanced frame, was_enhanced flag)
        """
        if frame is None or frame.size == 0:
            return frame, False

        should_enhance = force_night or (auto_detect and self.is_low_light(frame))
        if not should_enhance:
            return frame, False

        # Step 1: Suppress extreme headlight blooms first
        processed = self.suppress_headlight_glare(frame) if self.glare_suppression else frame

        # Step 2: Convert to LAB color space
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Step 3: Apply CLAHE on Luminance (L) channel
        cl = self.clahe.apply(l)

        # Step 4: Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([cl, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Slight unsharp mask to recover vehicle structural contours
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
        sharpened = cv2.addWeighted(enhanced_bgr, 1.25, gaussian, -0.25, 0)

        return sharpened, True


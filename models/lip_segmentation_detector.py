"""
Lip Segmentation-based Emotion Detection Module
================================================

This module provides lip segmentation using semantic segmentation models
as an alternative to MediaPipe landmark detection.

Uses BiSeNet face parsing model pre-trained on CelebAMask-HQ dataset.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings
import os
import threading

# Import BiSeNet from separate module
from .bisenet import BiSeNet


class LipSegmentationDetector:
    """
    Lip Segmentation-based Emotion Detection
    
    Uses semantic segmentation to extract lip regions and analyze their
    shape, position, and movements for emotion detection.
    
    Face parsing classes (CelebAMask-HQ):
    0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
    6: eye_g, 7: l_ear, 8: r_ear, 9: ear_r, 10: nose, 11: mouth,
    12: u_lip, 13: l_lip, 14: neck, 15: neck_l, 16: cloth, 17: hair, 18: hat
    """
    
    # Facial region class indices
    UPPER_LIP = 12
    LOWER_LIP = 13
    MOUTH = 11
    LEFT_EYE = 4
    RIGHT_EYE = 5
    LEFT_BROW = 2
    RIGHT_BROW = 3
    NOSE = 10
    SKIN = 1
    
    def __init__(self, model_path: Optional[str] = None, device: str = 'cpu'):
        """
        Initialize lip segmentation detector
        
        Args:
            model_path: Path to pre-trained BiSeNet weights (optional)
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = BiSeNet(n_classes=19).to(self.device)
        
        # Load pre-trained weights if available
        if model_path and os.path.exists(model_path):
            self.load_weights(model_path)
        else:
            warnings.warn("No pre-trained weights loaded. Model will need training or use random initialization.")
        
        self.model.eval()
        
        # JIT trace for faster inference (pre-allocates CUDA/CPU graph)
        try:
            dummy = torch.randn(1, 3, 512, 512).to(self.device)
            self.model = torch.jit.trace(self.model, dummy)
            print("  ✓ BiSeNet JIT traced for faster inference")
        except Exception as e:
            warnings.warn(f"JIT trace failed, using eager mode: {e}")
        
        # Pre-allocated input tensor buffer (avoids repeated allocation)
        self._input_buffer = torch.zeros(1, 3, 512, 512, dtype=torch.float32, device=self.device)
        
        # Async segmentation: persistent worker thread
        self._seg_lock = threading.Lock()
        self._latest_seg = None        # latest completed segmentation
        self._seg_event = threading.Event()
        self._seg_frame = None
        self._running = True
        self._seg_thread = threading.Thread(target=self._seg_worker, daemon=True)
        self._seg_thread.start()
        
        # Motion compensation: track face position when seg was computed
        self._seg_frame_gray = None    # grayscale of frame used for last seg
        self._warp_matrix = np.eye(2, 3, dtype=np.float32)  # identity
        
        # Cache for expensive cheek mask smoothing
        self._cached_cheek_soft = None
        self._cached_seg_id = None
        
        # Calibration data: per-emotion feature baselines
        # Keys: emotion name, Values: dict of feature averages
        self.calibration_profiles = {}  # e.g. {'happy': {'mouth_aspect': 3.1, ...}, ...}
        self.is_calibrated = False
        self.personalized_thresholds = {}  # filled by _compute_personalized_thresholds()
        
        # Face detector for preprocessing
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion classification based on lip features
        self.emotion_history = []
        self.history_size = 10
        
        # Normalization parameters (ImageNet stats)
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def load_weights(self, path: str):
        """Load pre-trained model weights"""
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded BiSeNet weights from {path}")
        except Exception as e:
            warnings.warn(f"Failed to load weights: {e}")
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess frame for BiSeNet into pre-allocated buffer (zero-copy).
        """
        # Convert BGR to RGB
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 512x512 (BiSeNet input size)
        img = cv2.resize(img, (512, 512))
        
        # Normalize in-place
        img = img.astype(np.float32) * (1.0 / 255.0)
        img -= self.mean
        img /= self.std
        
        # Write directly into pre-allocated buffer (avoids new tensor allocation)
        self._input_buffer[0] = torch.from_numpy(img.transpose(2, 0, 1))
        
        return self._input_buffer
    
    def segment_face(self, frame: np.ndarray) -> np.ndarray:
        """
        Perform face parsing segmentation (synchronous).
        """
        with torch.no_grad():
            img_tensor = self.preprocess_frame(frame)
            
            try:
                out = self.model(img_tensor)
                
                # Handle different model output formats
                if isinstance(out, tuple):
                    out = out[0]
                
                # Get segmentation prediction
                pred = out.squeeze(0).cpu().numpy()
                pred = pred.argmax(axis=0)
                
                # Resize back to original frame size
                pred = cv2.resize(pred.astype(np.uint8), (frame.shape[1], frame.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
                
                return pred
                
            except RuntimeError as e:
                raise RuntimeError(f"BiSeNet forward pass failed: {e}")
    
    def segment_face_async(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Non-blocking segmentation with motion compensation.
        
        Submits frame for background processing and returns the latest
        segmentation warped to match current face position using lightweight
        ECC-based or translation-based alignment.
        """
        # Always submit the latest frame (overwrites any pending frame)
        self._seg_frame = frame.copy()
        self._seg_event.set()  # wake worker
        
        # Return motion-compensated result
        with self._seg_lock:
            seg = self._latest_seg
            ref_gray = self._seg_frame_gray
        
        if seg is None:
            return None
        
        # Compute translation shift between reference frame and current frame
        if ref_gray is not None:
            try:
                cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscale for speed
                scale = 0.25
                small_ref = cv2.resize(ref_gray, None, fx=scale, fy=scale)
                small_cur = cv2.resize(cur_gray, None, fx=scale, fy=scale)
                
                # Phase correlation: fast sub-pixel translation estimation
                shift, _response = cv2.phaseCorrelate(
                    small_ref.astype(np.float64),
                    small_cur.astype(np.float64)
                )
                dx, dy = shift[0] / scale, shift[1] / scale
                
                # Only warp if shift is meaningful (> 1px) but not crazy (< 100px)
                if 1.0 < abs(dx) + abs(dy) < 100.0:
                    M = np.float32([[1, 0, dx], [0, 1, dy]])
                    warped = cv2.warpAffine(seg, M, (seg.shape[1], seg.shape[0]),
                                           flags=cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    return warped
            except Exception:
                pass  # fall through to unwarped
        
        return seg
    
    def _seg_worker(self):
        """Persistent background worker: waits for frames and runs segmentation."""
        while self._running:
            self._seg_event.wait()  # block until a frame is submitted
            self._seg_event.clear()
            
            frame = self._seg_frame
            if frame is None:
                continue
            
            try:
                result = self.segment_face(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self._seg_lock:
                    self._latest_seg = result
                    self._seg_frame_gray = gray
            except Exception:
                pass  # keep previous result on failure
    
    def extract_lip_features(self, segmentation_mask: np.ndarray) -> Dict:
        """
        Extract lip features from segmentation mask
        
        Args:
            segmentation_mask: Face parsing mask
            
        Returns:
            Dictionary of lip features
        """
        upper_lip_mask = (segmentation_mask == self.UPPER_LIP).astype(np.uint8)
        lower_lip_mask = (segmentation_mask == self.LOWER_LIP).astype(np.uint8)
        mouth_mask = (segmentation_mask == self.MOUTH).astype(np.uint8)
        
        # Combine lip regions
        lip_mask = np.maximum.reduce([upper_lip_mask, lower_lip_mask, mouth_mask])
        
        features = {}
        
        if lip_mask.sum() > 0:
            # Lip region properties
            contours, _ = cv2.findContours(lip_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Area
                features['lip_area'] = cv2.contourArea(largest_contour)
                
                # Bounding box
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['lip_bbox'] = (x, y, w, h)
                features['lip_aspect_ratio'] = w / (h + 1e-6)
                
                # Center of mass
                M = cv2.moments(largest_contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    features['lip_center'] = (cx, cy)
                
                # Convex hull and defects
                hull = cv2.convexHull(largest_contour, returnPoints=False)
                if len(largest_contour) > 3 and hull is not None and len(hull) > 0:
                    try:
                        defects = cv2.convexityDefects(largest_contour, hull)
                        features['convexity_defects'] = len(defects) if defects is not None else 0
                    except:
                        features['convexity_defects'] = 0
                
                # Openness (ratio of mouth area to total lip area)
                features['mouth_openness'] = mouth_mask.sum() / (lip_mask.sum() + 1e-6)
                
                # Upper vs lower lip ratio
                features['upper_lower_ratio'] = upper_lip_mask.sum() / (lower_lip_mask.sum() + 1e-6)
        
        return features
    
    def classify_emotion_from_lips(self, features: Dict) -> Tuple[str, float]:
        """
        Classify emotion based on lip features
        
        Args:
            features: Lip feature dictionary
            
        Returns:
            (emotion_label, confidence)
        """
        if not features or 'lip_aspect_ratio' not in features:
            return 'neutral', 0.5
        
        # Rule-based emotion classification
        # These are heuristics - can be replaced with ML classifier
        
        aspect_ratio = features['lip_aspect_ratio']
        openness = features.get('mouth_openness', 0)
        
        # Happy: Wide mouth (high aspect ratio), slight opening
        if aspect_ratio > 2.5 and openness < 0.3:
            return 'happy', 0.8
        
        # Surprised: Wide and open mouth
        if aspect_ratio > 2.0 and openness > 0.4:
            return 'surprise', 0.75
        
        # Angry/Disgust: Tight lips
        if aspect_ratio < 1.8 and openness < 0.15:
            return 'angry', 0.6
        
        # Sad: Drooping corners (would need corner detection)
        if aspect_ratio > 2.0 and openness < 0.2:
            return 'sad', 0.6
        
        # Default neutral
        return 'neutral', 0.5
    
    def extract_face_features(self, segmentation_mask: np.ndarray) -> Dict:
        """
        Extract geometric features from ALL segmented facial regions.
        
        Uses region areas, centroids, bounding boxes, and inter-region
        relationships to build a comprehensive feature vector for emotion
        classification based on FACS (Facial Action Coding System).
        
        Args:
            segmentation_mask: BiSeNet face parsing mask (H, W) with class ids 0-18
            
        Returns:
            Dictionary of facial features
        """
        features = {}
        h, w = segmentation_mask.shape
        face_area = (segmentation_mask > 0).sum() or 1  # total face pixels for normalization
        
        def region_stats(class_id):
            """Get area, centroid, and bbox for a region."""
            mask = (segmentation_mask == class_id).astype(np.uint8)
            area = int(mask.sum())
            if area < 5:
                return {'area': 0, 'cx': 0, 'cy': 0, 'w': 0, 'h': 0, 'aspect': 0}
            ys, xs = np.where(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            return {'area': area, 'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                    'aspect': bw / (bh + 1e-6), 'y_min': y0, 'y_max': y1,
                    'x_min': x0, 'x_max': x1}
        
        # Extract stats for each region
        l_eye = region_stats(self.LEFT_EYE)
        r_eye = region_stats(self.RIGHT_EYE)
        l_brow = region_stats(self.LEFT_BROW)
        r_brow = region_stats(self.RIGHT_BROW)
        nose = region_stats(self.NOSE)
        mouth = region_stats(self.MOUTH)
        u_lip = region_stats(self.UPPER_LIP)
        l_lip = region_stats(self.LOWER_LIP)
        
        # === MOUTH FEATURES ===
        # Combined lip mask for overall mouth metrics
        lip_mask = ((segmentation_mask == self.MOUTH) | 
                    (segmentation_mask == self.UPPER_LIP) | 
                    (segmentation_mask == self.LOWER_LIP)).astype(np.uint8)
        lip_area = int(lip_mask.sum())
        
        if lip_area > 5:
            ys, xs = np.where(lip_mask)
            mouth_width = int(xs.max() - xs.min() + 1)
            mouth_height = int(ys.max() - ys.min() + 1)
            features['mouth_width'] = mouth_width
            features['mouth_height'] = mouth_height
            features['mouth_aspect'] = mouth_width / (mouth_height + 1e-6)
            features['mouth_area_ratio'] = lip_area / face_area  # how much of face is mouth
            features['mouth_openness'] = mouth['area'] / (lip_area + 1e-6)  # inner mouth vs total
            features['upper_lower_lip_ratio'] = u_lip['area'] / (l_lip['area'] + 1e-6)
            features['mouth_cy'] = float(ys.mean())  # vertical position
        else:
            features['mouth_width'] = 0
            features['mouth_height'] = 0
            features['mouth_aspect'] = 1.5
            features['mouth_area_ratio'] = 0
            features['mouth_openness'] = 0
            features['upper_lower_lip_ratio'] = 1.0
            features['mouth_cy'] = h * 0.7
        
        # === EYE FEATURES ===
        avg_eye_area = (l_eye['area'] + r_eye['area']) / 2.0
        features['eye_area_ratio'] = avg_eye_area / face_area  # how open eyes are
        features['eye_aspect'] = (l_eye['aspect'] + r_eye['aspect']) / 2.0
        features['eye_asymmetry'] = abs(l_eye['area'] - r_eye['area']) / (avg_eye_area + 1e-6)
        
        # Eye openness: height of eye region (wider = more open)
        avg_eye_h = (l_eye['h'] + r_eye['h']) / 2.0
        avg_eye_w = (l_eye['w'] + r_eye['w']) / 2.0
        features['eye_openness'] = avg_eye_h / (avg_eye_w + 1e-6)  # height/width ratio
        
        # === EYEBROW FEATURES ===
        # Brow height: distance from brow centroid to eye centroid (larger = raised)
        if l_brow['area'] > 0 and l_eye['area'] > 0:
            l_brow_eye_dist = l_eye['cy'] - l_brow['cy']  # positive = brow above eye
        else:
            l_brow_eye_dist = 20.0
        if r_brow['area'] > 0 and r_eye['area'] > 0:
            r_brow_eye_dist = r_eye['cy'] - r_brow['cy']
        else:
            r_brow_eye_dist = 20.0
        
        features['brow_height'] = (l_brow_eye_dist + r_brow_eye_dist) / 2.0
        features['brow_height_asymmetry'] = abs(l_brow_eye_dist - r_brow_eye_dist)
        
        # Brow furrow: horizontal distance between inner edges of brows
        if l_brow['area'] > 0 and r_brow['area'] > 0:
            # Inner edges: right edge of left brow, left edge of right brow
            features['brow_furrow'] = r_brow.get('x_min', w//2) - l_brow.get('x_max', w//2)
        else:
            features['brow_furrow'] = 40.0
        
        # Brow area (thicker when furrowed/tense)
        features['brow_area_ratio'] = (l_brow['area'] + r_brow['area']) / (face_area + 1e-6)
        
        # === NOSE FEATURES ===
        features['nose_area_ratio'] = nose['area'] / face_area
        # Nose wrinkle indicator: nose area increases slightly when wrinkling
        
        # === INTER-REGION RELATIONSHIPS ===
        # Vertical face proportions normalized by nose centroid (stable reference)
        if nose['area'] > 0:
            nose_cy = nose['cy']
            # How far mouth center is below nose (opens more = further)
            features['nose_mouth_dist'] = features['mouth_cy'] - nose_cy
            # How far brows are above nose
            features['nose_brow_dist'] = nose_cy - (l_brow['cy'] + r_brow['cy']) / 2.0
        else:
            features['nose_mouth_dist'] = h * 0.15
            features['nose_brow_dist'] = h * 0.15
        
        # Face height estimate for normalization
        face_height = max(h * 0.5, features.get('nose_brow_dist', 50) + features.get('nose_mouth_dist', 50))
        features['face_height'] = face_height
        
        return features
    
    def classify_emotion_from_segmentation(self, features: Dict, 
                                           thresholds: Optional[Dict] = None) -> Tuple[str, float, np.ndarray]:
        """
        Classify emotion using FACS-inspired rules on segmentation features.
        
        All thresholds are adjustable via the `thresholds` dict.
        
        Args:
            features: Output of extract_face_features()
            thresholds: Optional dict to override default thresholds.
                Keys: 'sensitivity', 'happy_mouth_aspect', 'surprise_eye_open',
                      'angry_brow_furrow', 'sad_mouth_corner', 'neutral_floor'
                      
        Returns:
            (emotion_label, confidence, probabilities[7])
        """
        EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        probs = np.zeros(7)
        
        # Default thresholds (all adjustable)
        t = {
            'sensitivity': 1.3,          # global sensitivity multiplier
            'happy_mouth_aspect': 2.2,    # mouth width/height for smile
            'happy_openness_max': 0.35,   # mouth not gaping open
            'surprise_eye_open': 0.38,    # eye openness (h/w) threshold
            'surprise_brow_height': 25.0, # brow-eye distance for raised brows
            'surprise_mouth_open': 0.3,   # mouth openness (inner/total)
            'angry_brow_furrow': 30.0,    # inner brow distance (smaller = furrowed)
            'angry_brow_low': 18.0,       # brow height threshold (lower = angrier)
            'angry_eye_squint': 0.28,     # eye openness threshold (smaller = squinted)
            'sad_mouth_aspect': 2.5,      # narrow mouth
            'sad_mouth_open_max': 0.15,   # mouth mostly closed
            'sad_brow_inner_raise': 5.0,  # asymmetric brow (inner raised)
            'fear_eye_open': 0.35,        # wide eyes
            'fear_brow_height': 24.0,     # raised brows  
            'disgust_eye_squint': 0.26,   # squinted
            'disgust_nose_area': 0.045,   # wrinkled nose = larger area
            'neutral_floor': 0.12,        # below this, classify as neutral
        }
        if thresholds:
            t.update(thresholds)
        
        sens = t['sensitivity']
        
        # Extract features with defaults
        mouth_aspect = features.get('mouth_aspect', 1.5)
        mouth_openness = features.get('mouth_openness', 0)
        mouth_area_ratio = features.get('mouth_area_ratio', 0)
        eye_openness = features.get('eye_openness', 0.3)
        eye_area_ratio = features.get('eye_area_ratio', 0)
        eye_asymmetry = features.get('eye_asymmetry', 0)
        brow_height = features.get('brow_height', 20)
        brow_furrow = features.get('brow_furrow', 40)
        brow_height_asym = features.get('brow_height_asymmetry', 0)
        nose_area = features.get('nose_area_ratio', 0.03)
        upper_lower_ratio = features.get('upper_lower_lip_ratio', 1.0)
        face_height = features.get('face_height', 200)
        
        # Normalize key metrics by face height
        brow_height_n = brow_height / (face_height + 1e-6) * 100  # as percentage
        brow_furrow_n = brow_furrow / (face_height + 1e-6) * 100
        
        # ═══ HAPPY ═══
        # Wide mouth (smile), not gaping, possibly squinted eyes (crow's feet)
        happy = 0.0
        if mouth_aspect > t['happy_mouth_aspect']:
            happy += 0.3 * sens
        if mouth_aspect > t['happy_mouth_aspect'] * 1.3:
            happy += 0.2 * sens
        if mouth_openness < t['happy_openness_max']:
            happy += 0.15
        if mouth_area_ratio > 0.02:  # mouth takes up more face area when smiling
            happy += 0.15
        if eye_openness < 0.32:  # squinted from smiling
            happy += 0.1
        probs[3] = min(happy, 0.95)
        
        # ═══ SURPRISE ═══
        # Wide open eyes, raised brows, open mouth (O shape)
        surprise = 0.0
        if eye_openness > t['surprise_eye_open']:
            surprise += 0.25 * sens
        if eye_openness > t['surprise_eye_open'] * 1.2:
            surprise += 0.15 * sens
        if brow_height > t['surprise_brow_height']:
            surprise += 0.25 * sens
        if mouth_openness > t['surprise_mouth_open']:
            surprise += 0.2
        if mouth_aspect < 2.0 and mouth_openness > 0.25:  # O shape (round, not wide)
            surprise += 0.1
        probs[5] = min(surprise, 0.95)
        
        # ═══ ANGRY ═══
        # Furrowed brows (close together + lowered), tight lips, squinted
        angry = 0.0
        if brow_furrow < t['angry_brow_furrow']:
            angry += 0.25 * sens
        if brow_furrow < t['angry_brow_furrow'] * 0.7:
            angry += 0.15 * sens
        if brow_height < t['angry_brow_low']:
            angry += 0.2 * sens  
        if eye_openness < t['angry_eye_squint']:
            angry += 0.15
        if mouth_openness < 0.15 and mouth_aspect < 2.0:  # tight lips
            angry += 0.15
        probs[0] = min(angry, 0.95)
        
        # ═══ SAD ═══
        # Key indicators: mouth narrows/tightens, inner brows raise, eyes droop,
        # face compresses vertically (nose-mouth distance decreases)
        nose_mouth_dist = features.get('nose_mouth_dist', 50)
        sad = 0.0
        # Narrow, closed mouth (frown)
        if mouth_aspect < t['sad_mouth_aspect']:
            sad += 0.2 * sens
        if mouth_openness < t['sad_mouth_open_max']:
            sad += 0.15 * sens
        # Inner brow raise - classic sad AU1
        if brow_height_asym > t['sad_brow_inner_raise']:
            sad += 0.2 * sens
        # Droopy/partially closed eyes
        if eye_openness < 0.30:
            sad += 0.15 * sens
        # Lower lip pushes up (pout)
        if upper_lower_ratio > 1.2:
            sad += 0.1 * sens
        # Mouth area shrinks when frowning
        if mouth_area_ratio < 0.025:
            sad += 0.1 * sens
        # Face compresses vertically - nose to mouth distance decreases
        if nose_mouth_dist < face_height * 0.12:
            sad += 0.15 * sens
        # Lowered brow (not raised like surprise)
        if brow_height < 20:
            sad += 0.1
        probs[4] = min(sad, 0.95)
        
        # ═══ FEAR ═══
        # Wide eyes + raised brows + tense (not wide) mouth
        fear = 0.0
        if eye_openness > t['fear_eye_open']:
            fear += 0.25 * sens
        if brow_height > t['fear_brow_height']:
            fear += 0.2 * sens
        if brow_height_asym > 3.0:  # asymmetric brow raise
            fear += 0.1
        if mouth_openness > 0.15 and mouth_aspect < 2.2:  # tense open mouth
            fear += 0.2
        probs[2] = min(fear, 0.95)
        
        # ═══ DISGUST ═══
        # Squinted eyes, nose wrinkle (larger nose area), raised upper lip
        disgust = 0.0
        if eye_openness < t['disgust_eye_squint']:
            disgust += 0.2 * sens
        if nose_area > t['disgust_nose_area']:
            disgust += 0.25 * sens
        if upper_lower_ratio > 1.4:  # upper lip raised
            disgust += 0.2
        if mouth_aspect < 2.0:
            disgust += 0.1
        probs[1] = min(disgust, 0.95)
        
        # ═══ NEUTRAL ═══
        # Positive neutral: relaxed face — moderate mouth, eyes, brows all near resting position
        neutral = 0.0
        # Mouth at rest: moderate aspect ratio, low openness
        if 1.5 < mouth_aspect < 2.5 and mouth_openness < 0.2:
            neutral += 0.25
        # Eyes at rest: moderate openness (not wide, not squinted)
        if 0.25 < eye_openness < 0.38:
            neutral += 0.2
        # Brows at rest: moderate height, not furrowed
        if 15 < brow_height < 28 and brow_furrow > 25:
            neutral += 0.2
        # Low asymmetry
        if eye_asymmetry < 0.3 and brow_height_asym < 5:
            neutral += 0.15
        probs[6] = min(neutral, 0.9)
        
        # If no emotion (including neutral) is strong, boost neutral
        max_emotion = max(probs[:6])
        if max_emotion < t['neutral_floor'] and probs[6] < 0.3:
            probs[6] = 0.5
        
        # Normalize
        probs = probs / (probs.sum() + 1e-6)
        
        # Sharpen distribution
        temperature = 0.65
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / (probs.sum() + 1e-6)
        
        emotion_idx = np.argmax(probs)
        confidence = float(probs[emotion_idx])
        emotion = EMOTION_LABELS[emotion_idx]
        
        return emotion, confidence, probs
    
    # ═══════════════════════════════════════════════════
    # CALIBRATION SYSTEM
    # ═══════════════════════════════════════════════════
    
    CALIBRATION_EMOTIONS = ['neutral', 'happy', 'sad', 'angry', 'surprise', 'fear', 'disgust']
    
    # Which features matter most for each emotion (for weighted distance)
    EMOTION_KEY_FEATURES = {
        'happy':    ['mouth_aspect', 'mouth_openness', 'eye_openness', 'mouth_area_ratio'],
        'sad':      ['mouth_aspect', 'mouth_openness', 'brow_height', 'eye_openness', 'upper_lower_lip_ratio', 'nose_mouth_dist', 'mouth_area_ratio'],
        'angry':    ['brow_furrow', 'brow_height', 'eye_openness', 'mouth_aspect', 'mouth_openness'],
        'surprise': ['eye_openness', 'brow_height', 'mouth_openness', 'mouth_aspect'],
        'fear':     ['eye_openness', 'brow_height', 'mouth_openness', 'brow_height_asymmetry'],
        'disgust':  ['eye_openness', 'nose_area_ratio', 'upper_lower_lip_ratio', 'mouth_aspect'],
        'neutral':  ['mouth_aspect', 'eye_openness', 'brow_height', 'brow_furrow'],
    }
    
    def store_calibration(self, emotion: str, feature_samples: List[Dict]):
        """
        Store calibration data for an emotion from multiple captured samples.
        
        Args:
            emotion: Emotion label (e.g. 'happy', 'sad', etc.)
            feature_samples: List of feature dicts from extract_face_features()
        """
        if not feature_samples:
            return
        
        # Average all samples for this emotion
        avg_features = {}
        keys = feature_samples[0].keys()
        for key in keys:
            vals = [s[key] for s in feature_samples if key in s and isinstance(s[key], (int, float))]
            if vals:
                avg_features[key] = float(np.mean(vals))
        
        self.calibration_profiles[emotion] = avg_features
        
        # Mark as calibrated once we have at least neutral + 2 emotions
        if len(self.calibration_profiles) >= 3:
            self.is_calibrated = True
            self._compute_personalized_thresholds()
            print(f"  ✓ Calibrated {len(self.calibration_profiles)} emotions: {list(self.calibration_profiles.keys())}")
    
    def _compute_personalized_thresholds(self):
        """
        Compute personalized thresholds from calibration profiles.
        Uses neutral as baseline and sets thresholds at decision boundaries
        (midpoints between neutral and each emotion's features).
        """
        profiles = self.calibration_profiles
        neutral = profiles.get('neutral', {})
        if not neutral:
            return
        
        t = {}
        
        def midpoint(neutral_val, emotion_val, bias=0.5):
            """Compute threshold between neutral and emotion. bias 0.5=midpoint, >0.5=closer to emotion."""
            return neutral_val + bias * (emotion_val - neutral_val)
        
        # ── HAPPY thresholds ──
        if 'happy' in profiles:
            hp = profiles['happy']
            # User's neutral mouth_aspect vs happy → threshold just past neutral
            # Happy: wider mouth, so threshold between neutral and happy (closer to neutral = more sensitive)
            t['happy_mouth_aspect'] = midpoint(
                neutral.get('mouth_aspect', 2.0), hp.get('mouth_aspect', 3.5), 0.35)
            t['happy_openness_max'] = midpoint(
                neutral.get('mouth_openness', 0.1), hp.get('mouth_openness', 0.2), 1.5)  # allow higher
            print(f"    happy thresholds: mouth_aspect>{t['happy_mouth_aspect']:.2f} (neutral={neutral.get('mouth_aspect',0):.2f}, happy={hp.get('mouth_aspect',0):.2f})")
        
        # ── SURPRISE thresholds ──
        if 'surprise' in profiles:
            sp = profiles['surprise']
            t['surprise_eye_open'] = midpoint(
                neutral.get('eye_openness', 0.3), sp.get('eye_openness', 0.5), 0.4)
            t['surprise_brow_height'] = midpoint(
                neutral.get('brow_height', 20), sp.get('brow_height', 30), 0.4)
            t['surprise_mouth_open'] = midpoint(
                neutral.get('mouth_openness', 0.1), sp.get('mouth_openness', 0.4), 0.4)
        
        # ── ANGRY thresholds ──
        if 'angry' in profiles:
            ap = profiles['angry']
            # Angry: brows move closer (furrow decreases), brow lowers
            t['angry_brow_furrow'] = midpoint(
                neutral.get('brow_furrow', 40), ap.get('brow_furrow', 20), 0.4)
            t['angry_brow_low'] = midpoint(
                neutral.get('brow_height', 20), ap.get('brow_height', 15), 0.4)
            t['angry_eye_squint'] = midpoint(
                neutral.get('eye_openness', 0.3), ap.get('eye_openness', 0.22), 0.4)
        
        # ── SAD thresholds ──
        if 'sad' in profiles:
            sdp = profiles['sad']
            t['sad_mouth_aspect'] = midpoint(
                neutral.get('mouth_aspect', 2.0), sdp.get('mouth_aspect', 1.5), 0.4)
            t['sad_mouth_open_max'] = midpoint(
                neutral.get('mouth_openness', 0.1), sdp.get('mouth_openness', 0.05), 1.3)
            t['sad_brow_inner_raise'] = max(2.0, midpoint(
                neutral.get('brow_height_asymmetry', 2), sdp.get('brow_height_asymmetry', 8), 0.35))
        
        # ── FEAR thresholds ──
        if 'fear' in profiles:
            fp = profiles['fear']
            t['fear_eye_open'] = midpoint(
                neutral.get('eye_openness', 0.3), fp.get('eye_openness', 0.45), 0.4)
            t['fear_brow_height'] = midpoint(
                neutral.get('brow_height', 20), fp.get('brow_height', 28), 0.4)
        
        # ── DISGUST thresholds ──
        if 'disgust' in profiles:
            dp = profiles['disgust']
            t['disgust_eye_squint'] = midpoint(
                neutral.get('eye_openness', 0.3), dp.get('eye_openness', 0.2), 0.4)
            t['disgust_nose_area'] = midpoint(
                neutral.get('nose_area_ratio', 0.03), dp.get('nose_area_ratio', 0.05), 0.4)
        
        self.personalized_thresholds = t
        print(f"    Personalized thresholds: {t}")
    
    def classify_emotion_calibrated(self, features: Dict) -> Tuple[str, float, np.ndarray]:
        """
        Classify emotion using the rule-based classifier with personalized thresholds
        derived from calibration profiles.
        
        This feeds the calibration data directly into the rule-based system by
        adjusting thresholds (decision boundaries between neutral and each emotion).
        
        Args:
            features: Current frame's face features from extract_face_features()
            
        Returns:
            (emotion_label, confidence, probabilities[7])
        """
        # Use rule-based classifier with personalized thresholds
        thresholds = getattr(self, 'personalized_thresholds', {}).copy()
        thresholds['sensitivity'] = 1.3  # keep default sensitivity
        
        emotion, confidence, probs = self.classify_emotion_from_segmentation(features, thresholds)
        
        # Boost: penalize emotions that are far from their calibrated profile
        # and reward emotions close to their profile
        if self.calibration_profiles:
            EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            neutral = self.calibration_profiles.get('neutral', {})
            
            for emo, profile in self.calibration_profiles.items():
                idx = EMOTION_LABELS.index(emo)
                key_feats = self.EMOTION_KEY_FEATURES.get(emo, [])
                
                if emo == 'neutral':
                    # For neutral: boost if features are close to neutral baseline
                    deviation = 0.0
                    count = 0
                    for feat in key_feats:
                        if feat in features and feat in neutral:
                            # How far are we from neutral? Normalize by the neutral value
                            diff = abs(features[feat] - neutral[feat]) / (abs(neutral[feat]) + 1e-6)
                            deviation += diff
                            count += 1
                    if count > 0:
                        avg_dev = deviation / count
                        # Close to neutral (avg_dev < 0.15) → boost; far → leave alone
                        if avg_dev < 0.15:
                            probs[idx] *= 1.5
                        elif avg_dev < 0.25:
                            probs[idx] *= 1.2
                else:
                    # For non-neutral: check if key features moved TOWARD the calibrated direction
                    match_score = 0.0
                    count = 0
                    for feat in key_feats:
                        if feat in features and feat in profile and feat in neutral:
                            # Did the feature move in the right direction from neutral?
                            expected_delta = profile[feat] - neutral[feat]
                            actual_delta = features[feat] - neutral[feat]
                            if abs(expected_delta) > 1e-6:
                                # How much of the expected movement did we achieve?
                                ratio = actual_delta / expected_delta
                                match_score += max(0, min(ratio, 2.0))  # clamp 0-2
                                count += 1
                    if count > 0:
                        avg_match = match_score / count
                        # avg_match ~1.0 means features match calibration well
                        if avg_match > 0.6:
                            probs[idx] *= (1.0 + avg_match * 0.5)  # boost up to 2x
                        elif avg_match < 0.2:
                            probs[idx] *= 0.5  # penalize if wrong direction
            
            # Re-normalize and sharpen
            probs = probs / (probs.sum() + 1e-6)
            temperature = 0.55
            probs = np.power(probs, 1.0 / temperature)
            probs = probs / (probs.sum() + 1e-6)
            
            emotion_idx = np.argmax(probs)
            confidence = float(probs[emotion_idx])
            emotion = EMOTION_LABELS[emotion_idx]
        
        return emotion, confidence, probs
    
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process single frame and detect emotions
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Dictionary with detection results
        """
        # Segment face
        segmentation = self.segment_face(frame)
        
        # Extract lip features
        lip_features = self.extract_lip_features(segmentation)
        
        # Classify emotion
        emotion, confidence = self.classify_emotion_from_lips(lip_features)
        
        # Smooth with history
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.history_size:
            self.emotion_history.pop(0)
        
        # Most common emotion in history
        from collections import Counter
        emotion_counts = Counter(self.emotion_history)
        smoothed_emotion = emotion_counts.most_common(1)[0][0]
        
        return {
            'segmentation': segmentation,
            'lip_features': lip_features,
            'emotion': smoothed_emotion,
            'confidence': confidence,
            'raw_emotion': emotion
        }
    
    def visualize_segmentation(self, frame: np.ndarray, segmentation: np.ndarray, 
                              alpha: float = 0.6, lips_only: bool = False,
                              regions: str = 'lips_eyes') -> np.ndarray:
        """
        Overlay segmentation on frame
        
        Args:
            frame: Original frame
            segmentation: Segmentation mask
            alpha: Overlay transparency
            lips_only: Deprecated, use regions instead
            regions: 'lips' | 'lips_eyes' | 'all'
            
        Returns:
            Visualization frame
        """
        # Backward compat
        if lips_only:
            regions = 'lips'
        
        if regions == 'lips':
            colors = {
                self.UPPER_LIP: (255, 100, 100),    # Light red
                self.LOWER_LIP: (255, 150, 150),    # Pink  
                self.MOUTH: (200, 100, 255),        # Purple
            }
        elif regions in ('lips_eyes', 'lips_eyes_brows'):
            colors = {
                self.SKIN: (255, 220, 200),         # Soft peach (cheeks/face skin)
                self.UPPER_LIP: (255, 100, 100),    # Light red
                self.LOWER_LIP: (255, 150, 150),    # Pink  
                self.MOUTH: (200, 100, 255),        # Purple
                self.LEFT_EYE: (100, 200, 255),     # Light blue
                self.RIGHT_EYE: (100, 200, 255),    # Light blue
                self.LEFT_BROW: (180, 130, 255),    # Light purple
                self.RIGHT_BROW: (180, 130, 255),   # Light purple
                self.NOSE: (200, 255, 200),          # Light green
            }
        else:
            colors = {
                self.UPPER_LIP: (0, 0, 255),
                self.LOWER_LIP: (0, 100, 255),
                self.MOUTH: (0, 150, 255),
                self.LEFT_EYE: (255, 0, 0),
                self.RIGHT_EYE: (255, 0, 0),
                self.LEFT_BROW: (200, 100, 0),
                self.RIGHT_BROW: (200, 100, 0),
                self.NOSE: (0, 255, 0),
            }
        
        # Create colored overlay
        overlay = frame.copy()
        for class_id, color in colors.items():
            if class_id == self.SKIN:
                # Cheeks = skin pixels, excluding forehead and chin,
                # with buffer zone around eyes, nose, and mouth
                skin_mask = (segmentation == self.SKIN).copy()
                eye_mask = (segmentation == self.LEFT_EYE) | (segmentation == self.RIGHT_EYE)
                nose_mask = (segmentation == self.NOSE)
                mouth_mask = (segmentation == self.MOUTH) | (segmentation == self.UPPER_LIP) | (segmentation == self.LOWER_LIP)
                
                eye_rows = np.where(eye_mask.any(axis=1))[0]
                mouth_rows = np.where(mouth_mask.any(axis=1))[0]
                
                if len(eye_rows) > 0 and len(mouth_rows) > 0:
                    top = int(eye_rows.min())
                    bottom = int(mouth_rows.max())
                    
                    # Remove forehead and chin
                    skin_mask[:top, :] = False
                    skin_mask[bottom:, :] = False
                    
                    # Create buffer: dilate eyes/nose/mouth masks, subtract from skin
                    exclude = (eye_mask | nose_mask | mouth_mask).astype(np.uint8)
                    buf_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                    buffer_zone = cv2.dilate(exclude, buf_kernel, iterations=1).astype(bool)
                    skin_mask[buffer_zone] = False
                
                mask = skin_mask
            else:
                mask = (segmentation == class_id)
            overlay[mask] = color
        
        # Blend with original
        result = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)
        
        return result


# Import os for file checking
import os


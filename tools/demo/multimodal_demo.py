#!/usr/bin/env python3
"""
Multimodal Emotion Detection Demo with Tkinter
Real-time webcam with MediaPipe face mesh, emotion detection, and audio analysis.
Includes recording for training data collection.
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import tkinter as tk
from tkinter import ttk
import cv2
import torch
import numpy as np
from collections import deque
import time
from datetime import datetime
import threading
from typing import Optional, Dict, List, Tuple
from PIL import Image, ImageTk
import wave
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# MediaPipe - handle different versions
try:
    import mediapipe as mp
    # Test if solutions is available (newer versions)
    if hasattr(mp, 'solutions'):
        MEDIAPIPE_AVAILABLE = True
    else:
        # Try legacy import
        from mediapipe.python import solutions
        mp.solutions = solutions
        MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not available. Install with: pip install mediapipe")
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"Warning: MediaPipe error: {e}")

# FER (Facial Expression Recognition) - pretrained model
try:
    from fer.fer import FER
    FER_AVAILABLE = True
except ImportError:
    FER_AVAILABLE = False
    print("Warning: FER not available. Install with: pip install fer")

# Audio
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Warning: PyAudio not available. Install with: pip install pyaudio")

# Models
from models.emotion_detector import SpatialAttentionCNN, EMOTION_LABELS
from models.lip_segmentation_detector import LipSegmentationDetector


# Emotion colors (hex for Tkinter)
EMOTION_COLORS = {
    'angry': '#FF0000',       # Red
    'disgust': '#808000',     # Dark Yellow
    'fear': '#FF00FF',        # Magenta
    'happy': '#00FF00',       # Green
    'sad': '#0000FF',         # Blue
    'surprise': '#FFFF00',    # Yellow
    'neutral': '#808080'      # Gray
}

# BGR colors for OpenCV
EMOTION_COLORS_BGR = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 128),
    'fear': (255, 0, 255),
    'happy': (0, 255, 0),
    'sad': (255, 0, 0),
    'surprise': (0, 255, 255),
    'neutral': (128, 128, 128)
}


class VideoProcessor:
    """Handles video capture and processing."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.show_mask = True
        
        # Initialize lip segmentation detector
        self.lip_detector = None
        try:
            model_path = project_root / 'backbones' / 'face_parsing_models' / 'bisenet_celebamaskhq.pth'
            if model_path.exists():
                print("Loading BiSeNet lip segmentation model...")
                self.lip_detector = LipSegmentationDetector(
                    model_path=str(model_path),
                    device='cpu'
                )
                print("✓ Lip segmentation model loaded successfully")
            else:
                print(f"Warning: BiSeNet model not found at {model_path}")
                print("  Segmentation mask will not be available.")
        except Exception as e:
            print(f"Warning: Could not load lip segmentation model: {e}")
        
        # Baseline measurements for personalized calibration
        self.landmark_baseline = None
        self.baseline_measurements = []
        self.baseline_calibration_frames = 60  # Frames to collect baseline
        
        # Initialize ALL attributes FIRST (before any method calls)
        self.face_mesh = None
        self.mp_face_mesh = None
        self.fer_detector = None
        self.use_fer_model = False
        self.use_landmark_emotions = False
        self.use_seg_emotions = True  # Use segmentation-based emotion detection
        self.seg_emotion_thresholds = {'sensitivity': 1.3}  # Adjustable thresholds
        self.model = None
        self.fps_counter = deque(maxlen=30)
        self.emotion_history = deque(maxlen=15)
        
        # Feature smoothing: average raw features before classification
        self._feature_buffer = deque(maxlen=8)
        
        # Emotion stability: minimum consecutive frames before switching
        self._stable_emotion = 'neutral'
        self._stable_count = 0
        self._min_hold_frames = 5  # must see new emotion 5 times before switching
        
        # Performance optimization: frame skipping for expensive operations
        self.frame_counter = 0
        self.last_emotion_result = {
            'emotion': 'neutral',
            'confidence': 0.0,
            'probabilities': np.zeros(7)
        }
        self.emotion_process_interval = 3  # Process emotion every N frames
        self.cached_segmentation = None  # Cache segmentation results for better FPS
        
        # Calibration state
        self.calibrating = False
        self.calibration_emotion = None     # current emotion being calibrated
        self.calibration_samples = []       # feature samples for current emotion
        self.calibration_countdown = 0      # countdown timer
        self.calibration_capture = False    # actively capturing samples
        self.calibration_prompt = ""        # text to show on screen
        
        # Landmark regions (needed for visualization)
        self.landmark_regions = {
            'left_eye': list(range(133, 155)) + [33, 7, 163, 144, 145, 153, 154, 155, 133],
            'right_eye': list(range(362, 384)) + [263, 249, 390, 373, 374, 380, 381, 382, 362],
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'mouth': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82],
            'nose': [1, 2, 98, 327, 4, 5, 195, 197, 6]
        }
        
        # Face detector fallback
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Initialize MediaPipe if available
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self._init_face_mesh()
            except Exception as e:
                print(f"MediaPipe initialization failed: {e}")
                self._init_fer_only()
        else:
            print("MediaPipe not available, using FER model only")
            self._init_fer_only()
    
    def _init_fer_only(self):
        """Initialize FER model without MediaPipe"""
        # Landmark regions already initialized in __init__
        
        if FER_AVAILABLE:
            try:
                print("Loading pretrained FER model (MTCNN + CNN)...")
                self.fer_detector = FER(mtcnn=True)
                self.use_fer_model = True
                self.use_landmark_emotions = False
                print("✓ FER pretrained model loaded successfully")
                print("  - Face detection: MTCNN")
                print("  - Emotion model: Pretrained CNN")
            except Exception as e:
                print(f"Could not load FER model: {e}")
                print("Warning: No emotion detection available!")
                self.use_fer_model = False
                self.use_landmark_emotions = False
        else:
            print("Warning: No emotion detection libraries available")
            print("Install 'fer' with: pip install fer")
            self.use_fer_model = False
            self.use_landmark_emotions = False
    
    def _init_face_mesh(self):
        """Initialize or reinitialize MediaPipe FaceMesh with default parameters."""
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            self.face_mesh.close()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Use pretrained FER model by default (best accuracy)
        # Falls back to landmark-based if FER not available
        self.use_landmark_emotions = False  # Prefer pretrained model
        
        # Try to load pretrained FER model (automatic download)
        if FER_AVAILABLE:
            try:
                print("Loading pretrained FER model (MTCNN + CNN)...")
                # Use MTCNN for face detection (more accurate than OpenCV)
                self.fer_detector = FER(mtcnn=True)
                self.use_fer_model = True
                print("✓ FER pretrained model loaded successfully")
                print("  - Face detection: MTCNN")
                print("  - Emotion model: Pretrained CNN")
            except Exception as e:
                print(f"Could not load FER model: {e}")
                self.use_landmark_emotions = True
                self.use_fer_model = False
        else:
            print("FER library not available, using landmark-based detection")
            self.use_landmark_emotions = True
            self.use_fer_model = False
        
        self.model = None  # Not using custom model
        if self.use_fer_model:
            print("Using FER pretrained CNN for visual emotion detection")
        else:
            print("Using MediaPipe landmark-based emotion detection")
            print("  (Install 'fer' package for better accuracy)")
        
        # Face detector fallback, fps_counter, emotion_history, and landmark_regions
        # are all initialized in __init__ - no need to re-initialize here
    
    def _create_emotion_model(self):
        """Create emotion detection model with ResNet50 backbone."""
        from torchvision import models
        import torch.nn as nn
        
        # Use ResNet50 - better feature extraction than MobileNet
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace the final classification layer for 7 emotions
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 7)
        )
        
        return model.to(self.device)
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame for face detection and emotion recognition."""
        result = {
            'landmarks': None,
            'face_bbox': None,
            'emotion': 'neutral',
            'confidence': 0.0,
            'probabilities': np.zeros(7),
            'segmentation': None
        }
        
        # Performance optimization: skip expensive operations on some frames
        self.frame_counter += 1
        should_process_emotion = (self.frame_counter % self.emotion_process_interval == 0)
        
        # Segmentation runs async in background thread — never blocks the main loop
        if self.lip_detector is not None:
            seg = self.lip_detector.segment_face_async(frame)
            if seg is not None:
                result['segmentation'] = seg
                self.cached_segmentation = seg
            elif self.cached_segmentation is not None:
                result['segmentation'] = self.cached_segmentation
        
        # Calibration sample collection — runs every frame, only needs segmentation
        if (self.calibrating and self.calibration_capture
                and result['segmentation'] is not None and self.lip_detector is not None):
            cal_features = self.lip_detector.extract_face_features(result['segmentation'])
            if cal_features:  # ensure we got valid features
                self.calibration_samples.append(cal_features)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect face and landmarks with MediaPipe
        if self.face_mesh is not None:
            mp_results = self.face_mesh.process(frame_rgb)
            
            if mp_results.multi_face_landmarks:
                landmarks = mp_results.multi_face_landmarks[0]
                h, w = frame.shape[:2]
                
                # Extract landmark points
                landmark_points = []
                for lm in landmarks.landmark:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    z = lm.z
                    landmark_points.append([x, y, z])
                
                result['landmarks'] = np.array(landmark_points, dtype=np.float32)
                
                # Calculate face bounding box from landmarks
                x_coords = result['landmarks'][:, 0]
                y_coords = result['landmarks'][:, 1]
                padding = 20
                x_min = max(0, int(x_coords.min()) - padding)
                y_min = max(0, int(y_coords.min()) - padding)
                x_max = min(w, int(x_coords.max()) + padding)
                y_max = min(h, int(y_coords.max()) + padding)
                
                result['face_bbox'] = (x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Only run expensive emotion detection periodically
                if should_process_emotion:
                    # Choose emotion detection method
                    if self.use_fer_model and not self.use_landmark_emotions:
                        # Use pretrained FER model (best accuracy)
                        emotion, confidence, probs, fer_bbox = self.detect_emotion_fer(frame)
                        if emotion is None:
                            # FER failed, fallback to landmarks
                            emotion, confidence, probs = self.detect_emotion_from_landmarks(result['landmarks'], h)
                        elif fer_bbox is not None:
                            # Use FER's bounding box (more accurate)
                            result['face_bbox'] = fer_bbox
                    elif self.use_landmark_emotions:
                        # Use landmark-based emotion detection
                        emotion, confidence, probs = self.detect_emotion_from_landmarks(result['landmarks'], h)
                    else:
                        # Fallback
                        emotion, confidence, probs = 'neutral', 0.5, np.zeros(7)
                        probs[6] = 1.0
                    
                    # Fuse with segmentation-based emotion if available
                    if self.use_seg_emotions and result['segmentation'] is not None and self.lip_detector is not None:
                        seg_features = self.lip_detector.extract_face_features(result['segmentation'])
                        
                        # Smooth features before classification to reduce noise
                        self._feature_buffer.append(seg_features)
                        if len(self._feature_buffer) >= 2:
                            smoothed = {}
                            for key in seg_features:
                                vals = [f[key] for f in self._feature_buffer if key in f and isinstance(f[key], (int, float))]
                                if vals:
                                    smoothed[key] = float(np.mean(vals))
                            seg_features = smoothed
                        
                        # Use calibrated classifier if available, else rule-based
                        if self.lip_detector.is_calibrated:
                            seg_emotion, seg_conf, seg_probs = self.lip_detector.classify_emotion_calibrated(seg_features)
                            base_seg_weight = 0.3
                        else:
                            seg_emotion, seg_conf, seg_probs = self.lip_detector.classify_emotion_from_segmentation(
                                seg_features, self.seg_emotion_thresholds
                            )
                            base_seg_weight = 0.2
                        
                        # Confidence-adaptive fusion: if FER is very sure, trust it more
                        fer_conf = float(np.max(probs))
                        if fer_conf > 0.7:
                            seg_weight = base_seg_weight * 0.5  # FER very confident → barely use seg
                        elif fer_conf > 0.5:
                            seg_weight = base_seg_weight         # normal blend
                        else:
                            seg_weight = base_seg_weight * 1.5   # FER uncertain → lean more on seg
                        
                        # Only let seg override if it strongly agrees or FER is weak
                        if seg_emotion == emotion:
                            # Agreement — boost confidence
                            seg_weight = min(seg_weight * 1.3, 0.5)
                        
                        # Weighted fusion
                        probs = probs * (1 - seg_weight) + seg_probs * seg_weight
                        probs = probs / (probs.sum() + 1e-6)
                        emotion_idx = np.argmax(probs)
                        confidence = float(probs[emotion_idx])
                        emotion = EMOTION_LABELS[emotion_idx]
                    
                    # Cache result for next frames
                    self.last_emotion_result = {
                        'emotion': emotion,
                        'confidence': confidence,
                        'probabilities': probs
                    }
                    
                    # Smooth emotion predictions
                    self.emotion_history.append((emotion, confidence, probs))
                    emotion, confidence, probs = self._get_smoothed_emotion()
                else:
                    # Reuse last result to reduce CPU load
                    emotion = self.last_emotion_result['emotion']
                    confidence = self.last_emotion_result['confidence']
                    probs = self.last_emotion_result['probabilities']
                
                result['emotion'] = emotion
                result['confidence'] = confidence
                result['probabilities'] = probs
        else:
            # No MediaPipe - try FER which has its own face detector
            if self.use_fer_model:
                emotion, confidence, probs, fer_bbox = self.detect_emotion_fer(frame)
                if emotion is not None and fer_bbox is not None:
                    result['face_bbox'] = fer_bbox
                    result['emotion'] = emotion
                    result['confidence'] = confidence
                    result['probabilities'] = probs
            else:
                # Last fallback to Haar Cascade
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    result['face_bbox'] = (x, y, w, h)
                    # Try segmentation-based emotion even without landmarks
                    if self.use_seg_emotions and result['segmentation'] is not None and self.lip_detector is not None:
                        seg_features = self.lip_detector.extract_face_features(result['segmentation'])
                        emotion, confidence, probs = self.lip_detector.classify_emotion_from_segmentation(
                            seg_features, self.seg_emotion_thresholds
                        )
                        result['emotion'] = emotion
                        result['confidence'] = confidence
                        result['probabilities'] = probs
                    else:
                        probs = np.zeros(7)
                        probs[6] = 1.0
                        result['emotion'] = 'neutral'
                        result['confidence'] = 1.0
                        result['probabilities'] = probs
        
        return result
    
    def _get_smoothed_emotion(self) -> tuple:
        """Get smoothed emotion from history with strong hysteresis."""
        if len(self.emotion_history) == 0:
            return 'neutral', 0.5, np.zeros(7)
        
        # Exponential weighted average — recent frames weigh more
        all_probs = np.array([h[2] for h in self.emotion_history])
        n = len(all_probs)
        weights = np.exp(np.linspace(-1.5, 0, n))  # steeper: oldest=0.22, newest=1.0
        weights /= weights.sum()
        avg_probs = np.average(all_probs, axis=0, weights=weights)
        
        emotion_idx = np.argmax(avg_probs)
        confidence = avg_probs[emotion_idx]
        new_emotion = EMOTION_LABELS[emotion_idx]
        
        # Strong hysteresis: need 20% margin AND consecutive agreement
        if new_emotion != self._stable_emotion:
            cur_idx = EMOTION_LABELS.index(self._stable_emotion)
            cur_prob = avg_probs[cur_idx]
            margin = confidence - cur_prob
            
            if margin < 0.20:
                # Not enough margin — keep current
                return self._stable_emotion, cur_prob, avg_probs
            
            # Count how many of last N raw detections agree with new emotion
            recent = list(self.emotion_history)[-self._min_hold_frames:]
            agree_count = sum(1 for h in recent if EMOTION_LABELS[np.argmax(h[2])] == new_emotion)
            
            if agree_count < self._min_hold_frames:
                # Not enough consecutive agreement — keep current
                return self._stable_emotion, cur_prob, avg_probs
        
        # Switch accepted
        self._stable_emotion = new_emotion
        return new_emotion, confidence, avg_probs
    
    def detect_emotion_from_landmarks(self, landmarks: np.ndarray, face_height: float) -> tuple:
        """
        Detect emotion from facial landmarks using geometric features.
        Improved version with better calibration and balanced detection.
        """
        probs = np.zeros(7)  # angry, disgust, fear, happy, sad, surprise, neutral
        
        # Get key points
        def get_point(idx):
            if idx < len(landmarks):
                return landmarks[idx][:2]  # x, y only
            return np.array([0, 0])
        
        # === Calculate facial feature measurements ===
        
        # Mouth measurements
        mouth_left = get_point(61)   # Left corner
        mouth_right = get_point(291) # Right corner
        mouth_top = get_point(13)    # Top of upper lip
        mouth_bottom = get_point(14) # Bottom of lower lip
        upper_lip_top = get_point(0)
        lower_lip_bottom = get_point(17)
        
        mouth_width = np.linalg.norm(mouth_right - mouth_left)
        mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
        lip_opening = np.linalg.norm(lower_lip_bottom - upper_lip_top)
        
        # Mouth corner positions (for smile detection)
        mouth_center_y = (mouth_left[1] + mouth_right[1]) / 2
        nose_tip = get_point(4)
        
        # Check if mouth corners are raised (smile) relative to center
        mouth_corner_raise = nose_tip[1] - mouth_center_y  # Positive = corners raised
        
        # Eye measurements  
        left_eye_top = get_point(159)
        left_eye_bottom = get_point(145)
        left_eye_left = get_point(33)
        left_eye_right = get_point(133)
        
        right_eye_top = get_point(386)
        right_eye_bottom = get_point(374)
        right_eye_left = get_point(362)
        right_eye_right = get_point(263)
        
        left_eye_height = np.linalg.norm(left_eye_bottom - left_eye_top)
        left_eye_width = np.linalg.norm(left_eye_right - left_eye_left)
        right_eye_height = np.linalg.norm(right_eye_bottom - right_eye_top)
        right_eye_width = np.linalg.norm(right_eye_right - right_eye_left)
        
        # Eye aspect ratio (EAR) - lower = more closed
        left_ear = left_eye_height / (left_eye_width + 1e-6)
        right_ear = right_eye_height / (right_eye_width + 1e-6)
        avg_ear = (left_ear + right_ear) / 2
        
        # Eyebrow measurements
        left_brow_inner = get_point(107)
        left_brow_outer = get_point(70)
        right_brow_inner = get_point(336)
        right_brow_outer = get_point(300)
        
        # Brow height relative to eyes
        left_brow_height = left_eye_top[1] - left_brow_inner[1]
        right_brow_height = right_eye_top[1] - right_brow_inner[1]
        avg_brow_height = (left_brow_height + right_brow_height) / 2
        
        # Brow furrow (inner brows close together = angry/concentrated)
        brow_distance = np.linalg.norm(right_brow_inner - left_brow_inner)
        
        # === Normalize measurements by face size ===
        face_scale = face_height / 400.0  # Normalize to ~400px face
        
        mouth_width_n = mouth_width / (100 * face_scale)      # Typical ~100px
        mouth_height_n = mouth_height / (15 * face_scale)     # Typical ~15px  
        lip_opening_n = lip_opening / (40 * face_scale)       # Typical ~40px
        mouth_corner_n = mouth_corner_raise / (30 * face_scale)
        ear_n = avg_ear / 0.25                                # Typical EAR ~0.25
        brow_height_n = avg_brow_height / (25 * face_scale)   # Typical ~25px
        brow_dist_n = brow_distance / (60 * face_scale)       # Typical ~60px
        
        # === Emotion Detection Rules (with default thresholds) ===
        sens = 1.5  # Sensitivity multiplier
        
        # Apply sensitivity to normalized measurements (amplify deviations from 1.0)
        # This makes small expressions more detectable
        mouth_corner_n_adj = mouth_corner_n * sens
        mouth_width_n_adj = 1.0 + (mouth_width_n - 1.0) * sens
        ear_n_adj = 1.0 + (ear_n - 1.0) * sens
        brow_height_n_adj = 1.0 + (brow_height_n - 1.0) * sens
        brow_dist_n_adj = 1.0 + (brow_dist_n - 1.0) * sens
        lip_opening_n_adj = 1.0 + (lip_opening_n - 1.0) * sens
        
        # HAPPY: Raised mouth corners, wide mouth, possibly squinted eyes
        smile_score = 0.0
        if mouth_corner_n_adj > 0.15:  # Even slight corner raise
            smile_score += 0.35
        if mouth_corner_n_adj > 0.35:  # Clear smile
            smile_score += 0.2
        if mouth_width_n_adj > 1.02:   # Slightly wide mouth
            smile_score += 0.15
        if mouth_height_n > 0.6:  # Any mouth opening
            smile_score += 0.1
        if ear_n_adj < 0.98:  # Slightly squinted (crow's feet)
            smile_score += 0.1
        probs[3] = min(smile_score, 0.9)  # happy
        
        # SURPRISE: Wide eyes, raised brows, open mouth (O shape)
        surprise_score = 0.0
        if ear_n_adj > 1.08:  # Slightly wide eyes
            surprise_score += 0.3
        if ear_n_adj > 1.18:  # Very wide eyes
            surprise_score += 0.2
        if brow_height_n_adj > 1.05:  # Slightly raised eyebrows
            surprise_score += 0.25
        if lip_opening_n_adj > 1.1:  # Open mouth
            surprise_score += 0.2
        probs[5] = min(surprise_score, 0.9)  # surprise
        
        # ANGRY: Furrowed brows (close together, lowered), tight mouth, squinted
        angry_score = 0.0
        if brow_dist_n_adj < 0.95:  # Brows slightly close together
            angry_score += 0.25
        if brow_dist_n_adj < 0.85:  # Brows very close (furrowed)
            angry_score += 0.2
        if brow_height_n_adj < 0.95:  # Lowered brows
            angry_score += 0.2
        if ear_n_adj < 0.95:  # Squinted eyes
            angry_score += 0.15
        if mouth_corner_n_adj < -0.1:  # Slightly downturned mouth
            angry_score += 0.1
        probs[0] = min(angry_score, 0.9)  # angry
        
        # SAD: Lowered mouth corners, droopy eyes, inner brows raised
        sad_score = 0.0
        if mouth_corner_n_adj < -0.08:  # Slightly downturned corners
            sad_score += 0.3
        if mouth_corner_n_adj < -0.25:  # Clearly downturned
            sad_score += 0.2
        if ear_n_adj < 0.98:  # Slightly droopy look
            sad_score += 0.15
        if brow_height_n_adj > 0.95 and brow_height_n_adj < 1.15:  # Inner brow raise
            sad_score += 0.15
        probs[4] = min(sad_score, 0.9)  # sad
        
        # FEAR: Wide eyes, raised inner brows, tense mouth
        fear_score = 0.0
        if ear_n_adj > 1.03:  # Slightly wide eyes
            fear_score += 0.2
        if ear_n_adj > 1.12:  # Wide eyes
            fear_score += 0.2
        if brow_height_n_adj > 1.05:  # Raised brows
            fear_score += 0.2
        if mouth_height_n > 0.7 and mouth_width_n_adj < 1.05:  # Tense, slightly open
            fear_score += 0.15
        probs[2] = min(fear_score, 0.9)  # fear
        
        # DISGUST: Wrinkled nose area, raised upper lip, squinted
        disgust_score = 0.0
        if ear_n_adj < 0.85:  # Squinted
            disgust_score += 0.25
        if mouth_corner_n_adj < 0 and lip_opening_n_adj > 0.7:  # Asymmetric, lip raised
            disgust_score += 0.3
        if brow_dist_n_adj < 0.95 and ear_n_adj < 0.9:  # Furrowed + squinted
            disgust_score += 0.15
        probs[1] = min(disgust_score, 0.9)  # disgust
        
        # === Determine final emotion ===
        # Find the strongest non-neutral emotion
        max_emotion_score = max(probs[:6])
        
        # Neutral bias threshold
        neutral_thresh = 0.12
        if max_emotion_score < neutral_thresh:
            # Very weak - still mostly neutral but show some emotion
            probs[6] = 0.5
            probs[:6] *= 0.8  # Keep more of the emotion signal
        elif max_emotion_score < 0.3:
            # Weak emotion - balanced with neutral
            probs[6] = 0.35
        else:
            # Clear emotion detected - minimize neutral
            probs[6] = max(0.05, 0.25 - max_emotion_score)
        
        # Normalize to sum to 1
        probs = probs / (probs.sum() + 1e-6)
        
        # Apply slight sharpening to make the dominant emotion stand out
        temperature = 0.6  # Lower = sharper distribution
        probs = np.power(probs, 1.0 / temperature)
        probs = probs / (probs.sum() + 1e-6)
        
        emotion_idx = np.argmax(probs)
        confidence = probs[emotion_idx]
        emotion = EMOTION_LABELS[emotion_idx]
        
        return emotion, confidence, probs
    
    def detect_emotion(self, face_img: np.ndarray) -> tuple:
        """Detect emotion from face image using pretrained FER model."""
        
        # No model available, return neutral
        probs = np.zeros(7)
        probs[6] = 1.0
        return 'neutral', 1.0, probs
    
    def detect_emotion_fer(self, frame: np.ndarray) -> tuple:
        """Detect emotion using pretrained FER model on full frame."""
        if not self.use_fer_model or self.fer_detector is None:
            return None, 0.0, np.zeros(7), None
        
        try:
            # FER expects BGR frame and handles face detection internally
            result = self.fer_detector.detect_emotions(frame)
            
            if result and len(result) > 0:
                # Get first face (largest/most confident)
                face_result = result[0]
                bbox = face_result['box']  # (x, y, w, h)
                emotions = face_result['emotions']  # dict of emotion: score
                
                # Convert to our format
                # FER emotions: 'angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'
                probs = np.array([
                    emotions.get('angry', 0),
                    emotions.get('disgust', 0),
                    emotions.get('fear', 0),
                    emotions.get('happy', 0),
                    emotions.get('sad', 0),
                    emotions.get('surprise', 0),
                    emotions.get('neutral', 0)
                ])
                
                # Normalize (FER already gives normalized probabilities)
                probs = probs / (probs.sum() + 1e-6)
                
                emotion_idx = np.argmax(probs)
                confidence = probs[emotion_idx]
                emotion_label = EMOTION_LABELS[emotion_idx]
                
                return emotion_label, confidence, probs, bbox
            
        except Exception as e:
            print(f"FER detection error: {e}")
        
        return None, 0.0, np.zeros(7), None
    
    def draw_visualization(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """Draw visualization on frame."""
        display_frame = frame.copy()
        
        # Draw segmentation mask overlay if available and enabled
        if self.show_mask and result.get('segmentation') is not None and self.lip_detector is not None:
            try:
                display_frame = self.lip_detector.visualize_segmentation(
                    display_frame, 
                    result['segmentation'],
                    alpha=0.6,
                    regions='lips_eyes'
                )
            except Exception as e:
                # If visualization fails, log the error and continue without it
                print(f"Warning: Segmentation visualization failed: {e}")
                pass
        
        # Draw landmarks and mesh (kept for reference, can be removed if not using MediaPipe)
        if result['landmarks'] is not None and MEDIAPIPE_AVAILABLE:
            landmarks = result['landmarks']
            
            # Only draw mesh/landmarks if NOT showing segmentation mask
            if not self.show_mask:
                self.draw_face_mesh(display_frame, landmarks)
                self.draw_landmarks(display_frame, landmarks, result['emotion'])
        
        # Draw face bounding box and emotion
        if result['face_bbox'] is not None:
            x, y, w, h = result['face_bbox']
            emotion = result['emotion']
            confidence = result['confidence']
            color = EMOTION_COLORS_BGR.get(emotion, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw emotion label
            label = f"{emotion.upper()}: {confidence:.0%}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(display_frame, label, (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return display_frame
    
    def draw_face_mesh(self, frame: np.ndarray, landmarks: np.ndarray):
        """Draw face mesh connections."""
        connections = [
            # Face oval
            (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
            (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
            (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
            (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
            (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
            (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
            # Lips outer
            (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
            (314, 405), (405, 321), (321, 375), (375, 291), (291, 61),
        ]
        
        for start_idx, end_idx in connections:
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                pt1 = tuple(landmarks[start_idx][:2].astype(int))
                pt2 = tuple(landmarks[end_idx][:2].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 1)
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, emotion: str):
        """Draw facial landmarks with emotion-based highlighting."""
        highlight_regions = []
        if emotion in ['happy', 'sad', 'disgust']:
            highlight_regions.append('mouth')
        if emotion in ['surprise', 'fear', 'angry']:
            highlight_regions.extend(['left_eyebrow', 'right_eyebrow'])
        if emotion in ['surprise', 'fear']:
            highlight_regions.extend(['left_eye', 'right_eye'])
        
        for i, (x, y, z) in enumerate(landmarks):
            color = (0, 255, 0)  # Default green
            radius = 1
            
            for region_name in highlight_regions:
                if region_name in self.landmark_regions:
                    if i in self.landmark_regions[region_name]:
                        color = (0, 0, 255)  # Red for highlighted
                        radius = 2
                        break
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)


class AudioProcessor:
    """Handles audio capture and analysis with SER (Speech Emotion Recognition)."""
    
    # Audio settings
    SAMPLE_RATE = 16000  # 16kHz for SER model (target rate)
    CHUNK = 1024
    
    def __init__(self):
        self.running = False
        self.listening = False
        self.device_sample_rate = 16000  # Actual device rate (may differ)
        self.needs_resampling = False  # True if device rate != target rate
        self.callback = None
        self.status_callback = None
        self.mic_index = None
        self.vad_threshold = 0.01  # Adjustable sensitivity (lower = more sensitive)
        
        # Audio buffer for SER (2 seconds of audio)
        self.audio_buffer = deque(maxlen=int(self.SAMPLE_RATE * 2 / self.CHUNK) * self.CHUNK)
        self.raw_audio_buffer = []  # For recording
        
        # SER buffer for emotion detection (collect 2 seconds before inference)
        self.ser_buffer = deque(maxlen=int(self.SAMPLE_RATE * 2))
        self.ser_last_inference = 0
        self.ser_inference_interval = 1.0  # Run SER every 1 second
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = []  # Raw audio for saving
        
        # Volume tracking for UI meter
        self.current_volume = 0.0  # 0.0 to 1.0
        self.peak_volume = 0.0
        self.volume_callback = None  # Callback for UI updates
        
        # Load pretrained Speech Emotion Recognition model
        self.ser_model = None
        self.use_ser = False
        try:
            from models.speech_emotion_recognition import SpeechEmotionRecognizer
            print("Loading pretrained Speech Emotion Recognition (SER) model...")
            self.ser_model = SpeechEmotionRecognizer()
            # Check if model loaded successfully (emotion2vec uses .model attribute)
            if self.ser_model.model is not None:
                self.use_ser = True
                print("✓ SER model loaded successfully (emotion2vec+)")
            else:
                print("SER model not available - will return neutral")
        except Exception as e:
            print(f"Could not load SER model: {e}")
            import traceback
            traceback.print_exc()
        
        # Find and list available microphones
        self._find_microphones()
    
    def _find_microphones(self):
        """Find available microphones and select the best one."""
        if not PYAUDIO_AVAILABLE:
            print("PyAudio not available")
            return
        
        # Suppress ALSA/JACK error spam during device enumeration
        import os, ctypes
        os.environ['JACK_NO_START_SERVER'] = '1'
        os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
        try:
            ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                                   ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
            def py_error_handler(filename, line, function, err, fmt):
                pass
            c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
            asound = ctypes.cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
        except Exception:
            pass
        
        try:
            import pyaudio
            # Temporarily redirect C-level stderr (fd 2) to suppress JACK errors during PyAudio init
            import sys
            _stderr_fd = sys.stderr.fileno()
            _saved_stderr = os.dup(_stderr_fd)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, _stderr_fd)
            try:
                p = pyaudio.PyAudio()
            finally:
                os.dup2(_saved_stderr, _stderr_fd)
                os.close(_saved_stderr)
                os.close(_devnull)
            print("\n=== Available Audio Devices ===")
            
            input_devices = []
            pulse_device = None
            default_device = None
            device_rates = {}  # Store sample rates for each device
            
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if info['maxInputChannels'] > 0:
                    native_rate = int(info['defaultSampleRate'])
                    input_devices.append((i, info['name'], info))
                    device_rates[i] = native_rate
                    print(f"  [{i}] {info['name']} (inputs: {info['maxInputChannels']}, rate: {native_rate})")
                    
                    # Look for PulseAudio device (preferred for Docker)
                    name_lower = info['name'].lower()
                    if 'pulse' in name_lower:
                        pulse_device = i
                    elif 'default' in name_lower:
                        default_device = i
            
            p.terminate()
            
            # Priority: 1. PulseAudio device, 2. DMIC/Microphone, 3. Default device, 4. First non-monitor device
            selected_idx = None
            mic_device = None
            
            # First look for actual microphone devices (DMIC, ACP, or containing "mic")
            for idx, name, info in input_devices:
                name_lower = name.lower()
                if 'dmic' in name_lower or 'acp' in name_lower or 'microphone' in name_lower or name_lower.endswith(' mic'):
                    mic_device = idx
                    break
            
            # Helper to get device name by index
            def get_device_name(dev_idx):
                for idx, name, info in input_devices:
                    if idx == dev_idx:
                        return name
                return "Unknown"
            
            if pulse_device is not None:
                selected_idx = pulse_device
                print(f"  -> Selected PulseAudio device: [{pulse_device}] {get_device_name(pulse_device)}")
            elif mic_device is not None:
                selected_idx = mic_device
                print(f"  -> Selected microphone: [{selected_idx}] {get_device_name(selected_idx)}")
            elif default_device is not None:
                selected_idx = default_device
                print(f"  -> Selected default device: [{default_device}] {get_device_name(default_device)}")
            else:
                # Try to find a suitable microphone (prefer non-monitor devices)
                for idx, name, info in input_devices:
                    name_lower = name.lower()
                    # Skip monitor/loopback devices
                    if 'monitor' not in name_lower and 'loopback' not in name_lower:
                        selected_idx = idx
                        print(f"  -> Selected microphone: [{idx}] {name}")
                        break
                
                if selected_idx is None and input_devices:
                    # Fallback to first available input device
                    selected_idx = input_devices[0][0]
                    print(f"  -> Fallback to: [{selected_idx}] {input_devices[0][1]}")
            
            # Store selected device and its native sample rate
            if selected_idx is not None:
                self.mic_index = selected_idx
                self.device_sample_rate = device_rates.get(selected_idx, 16000)
                if self.device_sample_rate != self.SAMPLE_RATE:
                    self.needs_resampling = True
                    print(f"  ⚠️  Device rate ({self.device_sample_rate}Hz) differs from target ({self.SAMPLE_RATE}Hz)")
                    print(f"  -> Will resample audio for SER model")
            
            print("================================\n")
            
        except Exception as e:
            print(f"Error finding microphones: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self, callback, status_callback):
        """Start audio processing thread."""
        self.callback = callback
        self.status_callback = status_callback
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        
        # Start audio capture thread for SER
        if PYAUDIO_AVAILABLE:
            self.audio_capture_thread = threading.Thread(target=self._run_audio_capture, daemon=True)
            self.audio_capture_thread.start()
    
    def _run_audio_capture(self):
        """Capture audio for SER emotion detection using speech_recognition library."""
        try:
            import speech_recognition as sr
        except ImportError:
            print("⚠️  speech_recognition not installed. Audio disabled.")
            print("   Install with: pip install SpeechRecognition")
            return
        
        # Suppress ALSA/JACK noise before initializing audio
        import os
        os.environ['JACK_NO_START_SERVER'] = '1'
        os.environ['JACK_NO_AUDIO_RESERVATION'] = '1'
        
        # Redirect ALSA error output to /dev/null during init
        import ctypes
        try:
            ERROR_HANDLER_FUNC = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                                   ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p)
            def py_error_handler(filename, line, function, err, fmt):
                pass  # Suppress all ALSA errors
            c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
            asound = ctypes.cdll.LoadLibrary('libasound.so.2')
            asound.snd_lib_error_set_handler(c_error_handler)
            print("  ✓ ALSA error output suppressed")
        except Exception:
            pass  # Non-critical if suppression fails
        
        try:
            # Use speech_recognition library (handles PulseAudio/PyAudio/ALSA automatically)
            print(f"  -> Starting audio capture with speech_recognition library...")
            
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            
            # Create microphone instance — use discovered mic_index if available
            mic_kwargs = {'sample_rate': self.SAMPLE_RATE}
            if self.mic_index is not None:
                mic_kwargs['device_index'] = self.mic_index
                print(f"  -> Using device index {self.mic_index}")
            
            # Suppress C-level stderr during Microphone init (triggers JACK/ALSA enumeration)
            import sys as _sys
            _stderr_fd = _sys.stderr.fileno()
            _saved_stderr = os.dup(_stderr_fd)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, _stderr_fd)
            try:
                microphone = sr.Microphone(**mic_kwargs)
            finally:
                os.dup2(_saved_stderr, _stderr_fd)
                os.close(_saved_stderr)
                os.close(_devnull)
            
            print(f"  ✓ Audio capture initialized at {self.SAMPLE_RATE}Hz (mono, 16-bit)")
            print(f"  Using speech_recognition with system audio backend")
            
            # Keep microphone context open for the entire capture session
            # This avoids re-opening PyAudio streams every iteration (which triggers ALSA enumeration spam)
            with microphone as source:
                print("  Adjusting for ambient noise...")
                recognizer.adjust_for_ambient_noise(source, duration=1)
                print("  ✓ Ready")
                
                # Read directly from the PyAudio stream for continuous audio capture.
                # recognizer.listen() only returns audio when speech is detected,
                # which means silence → no volume updates → dead meter.
                stream = source.stream
                # The stream is opened at the sample_rate we requested (16kHz).
                # PulseAudio handles resampling from device native rate.
                self.needs_resampling = False
                
                while self.running:
                    if not self.listening:
                        time.sleep(0.05)
                        continue
                    
                    try:
                        # Read raw PCM from the PyAudio stream (blocking, returns CHUNK frames)
                        raw_data = stream.read(self.CHUNK)
                        audio_array = np.frombuffer(raw_data, dtype=np.int16)
                        
                        if len(audio_array) == 0:
                            continue
                        
                        # Calculate volume level (RMS normalized to 0-1)
                        rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                        self.current_volume = min(1.0, rms / 10000.0)
                        self.peak_volume = max(self.peak_volume * 0.95, self.current_volume)
                        
                        # Send volume update to UI
                        if self.volume_callback:
                            self.volume_callback(self.current_volume, self.peak_volume)
                        
                        # Use original for general buffer
                        self.audio_buffer.extend(audio_array)
                        
                        # Use for SER buffer (already at 16kHz)
                        self.ser_buffer.extend(audio_array)
                        
                        if self.is_recording:
                            self.recorded_audio.extend(audio_array.tolist())
                        
                        # Run SER inference periodically
                        current_time = time.time()
                        if (self.use_ser and 
                            len(self.ser_buffer) >= self.SAMPLE_RATE * 2 and
                            current_time - self.ser_last_inference >= self.ser_inference_interval):
                            self._run_ser_inference()
                            self.ser_last_inference = current_time
                    
                    except IOError as e:
                        # Buffer overflow or underflow, skip
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"Audio capture error: {e}")
                        time.sleep(0.1)
            
            print("  Audio capture stopped")
            
        except Exception as e:
            print(f"Failed to start audio capture: {e}")
            import traceback
            traceback.print_exc()
    
    def _resample_audio(self, audio_array, target_length):
        """Resample audio to target length using linear interpolation."""
        try:
            from scipy import signal
            # Use scipy for high-quality resampling
            resampled = signal.resample(audio_array, target_length)
            return resampled.astype(np.int16)
        except ImportError:
            # Fallback to simple linear interpolation if scipy not available
            indices = np.linspace(0, len(audio_array) - 1, target_length)
            resampled = np.interp(indices, np.arange(len(audio_array)), audio_array)
            return resampled.astype(np.int16)
    
    def _run_ser_inference(self):
        """Run Speech Emotion Recognition on buffered audio."""
        if not self.use_ser or self.ser_model is None:
            return
        
        try:
            # Convert buffer to numpy array
            audio_data = np.array(list(self.ser_buffer), dtype=np.int16)
            
            # Convert to float for RMS calculation
            audio_float = audio_data.astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float ** 2))
            
            # Voice Activity Detection: Only process if RMS is above threshold
            # Typical speech RMS is 0.05-0.3, noise is usually < 0.02
            # Use adjustable threshold from UI
            
            if rms < self.vad_threshold:
                print(f"[SER] Skipping inference - audio too quiet (RMS: {rms:.4f} < {self.vad_threshold})")
                return
            
            print(f"[SER] Running inference on {len(audio_data)} samples (RMS: {rms:.4f})...")
            
            # Run SER prediction
            emotion, confidence, probs = self.ser_model.predict(audio_data, self.SAMPLE_RATE)
            
            print(f"[SER] Result: {emotion} ({confidence:.2%})")
            
            # Report all SER results (lower threshold for display)
            # Create emotion scores dict for callback
            emotion_scores = {
                'angry': float(probs[0]),
                'disgust': float(probs[1]),
                'fear': float(probs[2]),
                'happy': float(probs[3]),
                'sad': float(probs[4]),
                'surprise': float(probs[5]),
                'neutral': float(probs[6])
            }
            
            # Send SER result through callback (always send, let UI decide)
            if self.callback:
                print(f"[SER] Sending callback with {emotion}")
                self.callback({
                    'text': f"[SER: {emotion} {confidence:.0%}]",
                    'emotion_scores': emotion_scores,
                    'is_ser': True,
                    'is_partial': False
                })
                    
        except Exception as e:
            print(f"SER inference error: {e}")
            import traceback
            traceback.print_exc()
    
    def get_audio_waveform_for_training(self) -> Optional[np.ndarray]:
        """
        Get raw audio waveform suitable for training the acoustic model.
        Returns: numpy array of shape (samples,) at 16kHz
        """
        if len(self.audio_buffer) == 0:
            return None
        
        audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
        audio_data = audio_data / 32768.0  # Normalize to [-1, 1]
        return audio_data
    
    def start_recording(self):
        """Start recording audio for training data."""
        self.is_recording = True
        self.recorded_audio = []
        print("Audio recording started")
    
    def stop_recording(self) -> np.ndarray:
        """Stop recording and return recorded audio data."""
        self.is_recording = False
        audio_data = np.array(self.recorded_audio, dtype=np.float32) / 32768.0
        print(f"Audio recording stopped. Samples: {len(audio_data)}")
        return audio_data
    
    def _run(self):
        """Main audio thread - just keeps running for SER processing."""
        if not PYAUDIO_AVAILABLE:
            if self.status_callback:
                self.status_callback("Audio not available")
            return
        
        if self.status_callback:
            self.status_callback("SER Ready - Listening...")
        
        # Just keep the thread alive - actual audio capture happens in _run_audio_capture
        while self.running:
            if not self.listening:
                time.sleep(0.1)
                continue
            time.sleep(0.1)
    
    def start_listening(self):
        self.listening = True
    
    def stop_listening(self):
        self.listening = False
    
    def stop(self):
        self.running = False
        self.listening = False


class EmotionDemoApp:
    """Main application class using Tkinter with recording for training data collection."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Multimodal Emotion Detection Demo")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1a1a2e')
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()
        
        # State
        self.cap = None
        self.running = False
        self.current_emotion = 'neutral'
        self.current_confidence = 0.0
        self.current_probs = np.zeros(7)
        self.fps = 0.0
        self.fps_counter = deque(maxlen=30)
        
        # SER (Speech Emotion) state for fusion
        self.ser_probs = np.zeros(7)  # Latest SER probabilities
        self.ser_emotion = 'neutral'
        self.ser_confidence = 0.0
        self.ser_timestamp = 0.0  # When SER was last updated
        self.fusion_weight_audio = 0.4  # Weight for audio in fusion (0.4 audio, 0.6 video)
        
        # Recording state for training data
        self.is_recording_training_data = False
        self.recording_start_time = None
        self.recorded_frames = []         # Video frames (B, T, 3, H, W) format
        self.recorded_display_frames = [] # Display frames with mask overlay
        self.recorded_landmarks = []      # Landmarks (B, T, 468*3) format  
        self.recorded_emotions = []       # Emotion labels with timestamps
        self.recorded_audio_emotions = [] # Audio emotion results with timestamps
        self.training_data_dir = Path(project_root) / 'data' / 'training_recordings'
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup UI
        self.setup_ui()
        
        # Bind close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        """Setup the user interface."""
        # Configure styles
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#1a1a2e')
        style.configure('TLabel', background='#1a1a2e', foreground='#eee')
        style.configure('TButton', background='#16213e', foreground='white')
        style.configure('TCheckbutton', background='#1a1a2e', foreground='#eee')
        
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Video
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Video canvas
        self.video_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black', highlightthickness=2, highlightbackground='#333')
        self.video_canvas.pack(pady=10)
        
        # Video controls
        controls_frame = ttk.Frame(left_frame)
        controls_frame.pack(fill=tk.X, pady=5)
        
        self.start_btn = tk.Button(controls_frame, text="▶ Start Camera", command=self.toggle_camera,
                                   bg='#16213e', fg='white', font=('Arial', 10, 'bold'),
                                   activebackground='#e94560', padx=15, pady=5)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.mask_var = tk.BooleanVar(value=True)
        mask_cb = tk.Checkbutton(controls_frame, text="Show Mask", variable=self.mask_var,
                                 command=self.toggle_mask, bg='#1a1a2e', fg='#eee',
                                 selectcolor='#16213e', activebackground='#1a1a2e')
        mask_cb.pack(side=tk.LEFT, padx=10)
        
        self.calibrate_btn = tk.Button(controls_frame, text="🎯 Calibrate", command=self.start_calibration,
                                       bg='#16213e', fg='#0ff', font=('Arial', 10, 'bold'),
                                       activebackground='#e94560', padx=15, pady=5)
        self.calibrate_btn.pack(side=tk.LEFT, padx=5)
        
        # Right panel - Analysis
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        
        # Current emotion display
        emotion_frame = tk.LabelFrame(right_frame, text="Current Emotion", bg='#1a1a2e', fg='#eee',
                                      font=('Arial', 10, 'bold'))
        emotion_frame.pack(fill=tk.X, pady=5)
        
        self.emotion_label = tk.Label(emotion_frame, text="NEUTRAL", font=('Arial', 28, 'bold'),
                                      bg='#1a1a2e', fg='#808080')
        self.emotion_label.pack(pady=10)
        
        self.confidence_label = tk.Label(emotion_frame, text="Confidence: 0%", font=('Arial', 12),
                                         bg='#1a1a2e', fg='#aaa')
        self.confidence_label.pack(pady=5)
        
        # Emotion probabilities
        probs_frame = tk.LabelFrame(right_frame, text="Emotion Probabilities", bg='#1a1a2e', fg='#eee',
                                    font=('Arial', 10, 'bold'))
        probs_frame.pack(fill=tk.X, pady=5)
        
        self.prob_bars = {}
        self.prob_labels = {}
        for emotion in EMOTION_LABELS:
            row_frame = ttk.Frame(probs_frame)
            row_frame.pack(fill=tk.X, padx=5, pady=2)
            
            label = tk.Label(row_frame, text=emotion.capitalize(), width=10, anchor='w',
                           bg='#1a1a2e', fg=EMOTION_COLORS[emotion], font=('Arial', 9, 'bold'))
            label.pack(side=tk.LEFT)
            
            # Progress bar using canvas
            bar_canvas = tk.Canvas(row_frame, width=200, height=20, bg='#222', highlightthickness=1,
                                  highlightbackground='#333')
            bar_canvas.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            
            # Create bar rectangle
            bar = bar_canvas.create_rectangle(0, 0, 0, 20, fill=EMOTION_COLORS[emotion], outline='')
            
            # Value label
            value_label = tk.Label(row_frame, text="0%", width=5, bg='#1a1a2e', fg='#eee',
                                  font=('Arial', 9))
            value_label.pack(side=tk.RIGHT)
            
            self.prob_bars[emotion] = (bar_canvas, bar)
            self.prob_labels[emotion] = value_label
        
        # Audio controls
        audio_frame = tk.LabelFrame(right_frame, text="Audio Analysis", bg='#1a1a2e', fg='#eee',
                                    font=('Arial', 10, 'bold'))
        audio_frame.pack(fill=tk.X, pady=5)
        
        self.audio_btn = tk.Button(audio_frame, text="🎤 Start Audio", command=self.toggle_audio,
                                   bg='#16213e', fg='white', font=('Arial', 10, 'bold'),
                                   activebackground='#e94560', padx=15, pady=5)
        if not PYAUDIO_AVAILABLE:
            self.audio_btn.config(state=tk.DISABLED)
        self.audio_btn.pack(pady=5)
        
        self.audio_status_label = tk.Label(audio_frame, text="Audio: Idle", bg='#1a1a2e', fg='#888',
                                          font=('Arial', 9))
        self.audio_status_label.pack()
        
        # Microphone volume meter
        mic_vol_frame = ttk.Frame(audio_frame)
        mic_vol_frame.pack(fill=tk.X, padx=5, pady=5)
        
        mic_vol_label = tk.Label(mic_vol_frame, text="🎤 Mic Level:", bg='#1a1a2e', fg='#aaa',
                                 font=('Arial', 9))
        mic_vol_label.pack(side=tk.LEFT)
        
        # Volume bar canvas
        self.mic_vol_canvas = tk.Canvas(mic_vol_frame, width=180, height=20, bg='#222', 
                                        highlightthickness=1, highlightbackground='#333')
        self.mic_vol_canvas.pack(side=tk.LEFT, padx=5)
        
        # Create volume bar (green) and peak indicator (yellow)
        self.mic_vol_bar = self.mic_vol_canvas.create_rectangle(0, 0, 0, 20, fill='#00ff00', outline='')
        self.mic_peak_line = self.mic_vol_canvas.create_line(0, 0, 0, 20, fill='#ffff00', width=2)
        
        # Volume percentage label
        self.mic_vol_pct_label = tk.Label(mic_vol_frame, text="0%", width=4, bg='#1a1a2e', fg='#0f0',
                                          font=('Arial', 9, 'bold'))
        self.mic_vol_pct_label.pack(side=tk.LEFT)
        
        # SER status label (shows current speech emotion)
        self.ser_status_label = tk.Label(audio_frame, text="SER: Waiting for audio...", bg='#1a1a2e', fg='#0ff',
                                         font=('Arial', 10, 'bold'), wraplength=270)
        self.ser_status_label.pack(pady=5)
        
        # Microphone sensitivity slider
        mic_sens_frame = ttk.Frame(audio_frame)
        mic_sens_frame.pack(fill=tk.X, padx=5, pady=5)
        
        mic_sens_label = tk.Label(mic_sens_frame, text="🎙️ Mic Sensitivity:", bg='#1a1a2e', fg='#aaa',
                                   font=('Arial', 9))
        mic_sens_label.pack(side=tk.LEFT)
        
        # Sensitivity slider (lower = more sensitive)
        self.mic_sensitivity_var = tk.DoubleVar(value=0.05)
        mic_sens_slider = tk.Scale(mic_sens_frame, from_=0.01, to=0.15, orient=tk.HORIZONTAL,
                                   variable=self.mic_sensitivity_var, resolution=0.01,
                                   bg='#16213e', fg='#fff', troughcolor='#333',
                                   highlightthickness=0, length=150, width=10,
                                   command=self.on_sensitivity_change)
        mic_sens_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Sensitivity value label
        self.mic_sens_pct_label = tk.Label(mic_sens_frame, text="0.05", width=4, bg='#1a1a2e', fg='#0ff',
                                           font=('Arial', 9, 'bold'))
        self.mic_sens_pct_label.pack(side=tk.LEFT)
        
        # Emotion detection sensitivity slider
        emo_sens_frame = ttk.Frame(audio_frame)
        emo_sens_frame.pack(fill=tk.X, padx=5, pady=5)
        
        emo_sens_label = tk.Label(emo_sens_frame, text="😊 Emotion Sensitivity:", bg='#1a1a2e', fg='#aaa',
                                   font=('Arial', 9))
        emo_sens_label.pack(side=tk.LEFT)
        
        self.emo_sensitivity_var = tk.DoubleVar(value=1.3)
        emo_sens_slider = tk.Scale(emo_sens_frame, from_=0.5, to=3.0, orient=tk.HORIZONTAL,
                                   variable=self.emo_sensitivity_var, resolution=0.1,
                                   bg='#16213e', fg='#fff', troughcolor='#333',
                                   highlightthickness=0, length=150, width=10,
                                   command=self.on_emotion_sensitivity_change)
        emo_sens_slider.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.emo_sens_pct_label = tk.Label(emo_sens_frame, text="1.3", width=4, bg='#1a1a2e', fg='#0ff',
                                           font=('Arial', 9, 'bold'))
        self.emo_sens_pct_label.pack(side=tk.LEFT)
        
        # ===== Recording Controls for Training Data =====
        recording_frame = tk.LabelFrame(right_frame, text="📹 Training Data Recording", bg='#1a1a2e', fg='#ff0',
                                        font=('Arial', 10, 'bold'))
        recording_frame.pack(fill=tk.X, pady=5)
        
        rec_btn_frame = ttk.Frame(recording_frame)
        rec_btn_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.record_btn = tk.Button(rec_btn_frame, text="⏺ Start Recording", command=self.toggle_recording,
                                    bg='#16213e', fg='white', font=('Arial', 10, 'bold'),
                                    activebackground='#e94560', padx=15, pady=5)
        self.record_btn.pack(side=tk.LEFT, padx=5)
        
        self.recording_status_label = tk.Label(rec_btn_frame, text="Not Recording", bg='#1a1a2e', fg='#888',
                                               font=('Arial', 9))
        self.recording_status_label.pack(side=tk.LEFT, padx=10)
        
        # Recording info label
        rec_info_label = tk.Label(recording_frame, 
                                  text="Records: Video frames, landmarks,\naudio (16kHz), emotions",
                                  bg='#1a1a2e', fg='#666', font=('Arial', 8), justify=tk.LEFT)
        rec_info_label.pack(anchor='w', padx=5)
        
        # Recording stats
        self.recording_stats_label = tk.Label(recording_frame, text="Frames: 0 | Duration: 0.0s",
                                              bg='#1a1a2e', fg='#aaa', font=('Arial', 8))
        self.recording_stats_label.pack(anchor='w', padx=5, pady=2)
        
        # Status bar
        status_frame = tk.LabelFrame(right_frame, text="Status", bg='#1a1a2e', fg='#eee',
                                     font=('Arial', 10, 'bold'))
        status_frame.pack(fill=tk.X, pady=5)
        
        status_inner = ttk.Frame(status_frame)
        status_inner.pack(fill=tk.X, padx=5, pady=5)
        
        self.fps_label = tk.Label(status_inner, text="FPS: 0.0", bg='#1a1a2e', fg='#0f0',
                                 font=('Arial', 9, 'bold'))
        self.fps_label.pack(side=tk.LEFT, padx=10)
        
        mp_status = "✓" if MEDIAPIPE_AVAILABLE else "✗"
        mp_color = "#0f0" if MEDIAPIPE_AVAILABLE else "#f00"
        self.mp_label = tk.Label(status_inner, text=f"MediaPipe: {mp_status}", bg='#1a1a2e',
                                fg=mp_color, font=('Arial', 9))
        self.mp_label.pack(side=tk.LEFT, padx=10)
        
        audio_status = "✓" if PYAUDIO_AVAILABLE else "✗"
        audio_color = "#0f0" if PYAUDIO_AVAILABLE else "#f00"
        self.audio_avail_label = tk.Label(status_inner, text=f"Audio: {audio_status}", bg='#1a1a2e',
                                         fg=audio_color, font=('Arial', 9))
        self.audio_avail_label.pack(side=tk.LEFT, padx=10)
        
        # Bottom status
        self.status_label = tk.Label(self.root, text="Ready - Click 'Start Camera' to begin",
                                    bg='#16213e', fg='#eee', anchor='w', padx=10)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    

    
    def toggle_camera(self):
        """Start or stop the camera."""
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.status_label.config(text="Error: Could not open webcam")
                return
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            self.running = True
            self.start_btn.config(text="⏹ Stop Camera", bg='#e94560')
            self.status_label.config(text="Camera started")
            self.update_frame()
        else:
            self.running = False
            if self.cap:
                self.cap.release()
                self.cap = None
            self.start_btn.config(text="▶ Start Camera", bg='#16213e')
            self.video_canvas.delete("all")
            self.status_label.config(text="Camera stopped")
    
    def toggle_mask(self):
        """Toggle mask display."""
        self.video_processor.show_mask = self.mask_var.get()
    
    def on_sensitivity_change(self, value):
        """Update microphone sensitivity threshold."""
        sensitivity = float(value)
        self.audio_processor.vad_threshold = sensitivity
        self.mic_sens_pct_label.config(text=f"{sensitivity:.2f}")
    
    def on_emotion_sensitivity_change(self, value):
        """Update segmentation emotion detection sensitivity."""
        sensitivity = float(value)
        self.video_processor.seg_emotion_thresholds['sensitivity'] = sensitivity
        self.emo_sens_pct_label.config(text=f"{sensitivity:.1f}")
    
    def start_calibration(self):
        """Start the calibration sequence — guides user through each emotion."""
        if not self.running:
            self.status_label.config(text="Start camera first before calibrating")
            return
        if self.video_processor.lip_detector is None:
            self.status_label.config(text="Segmentation model not loaded")
            return
        
        self.calibrate_btn.config(state=tk.DISABLED, text="🎯 Calibrating...")
        self.video_processor.calibrating = True
        self._calibration_queue = list(self.video_processor.lip_detector.CALIBRATION_EMOTIONS)
        self._calibrate_next_emotion()
    
    def _calibrate_next_emotion(self):
        """Move to the next emotion in the calibration sequence."""
        if not self._calibration_queue:
            # Done!
            self.video_processor.calibrating = False
            self.video_processor.calibration_prompt = ""
            self.video_processor.calibration_capture = False
            # Clear emotion history so old pre-calibration predictions don't linger
            self.video_processor.emotion_history.clear()
            self.video_processor._feature_buffer.clear()
            self.video_processor._stable_emotion = 'neutral'
            self.calibrate_btn.config(state=tk.NORMAL, text="🎯 Recalibrate")
            n = len(self.video_processor.lip_detector.calibration_profiles)
            cal_status = "calibrated" if self.video_processor.lip_detector.is_calibrated else "NOT calibrated (need ≥3)"
            self.status_label.config(text=f"✓ Calibration complete — {n} emotions profiled ({cal_status})")
            print(f"  [calibration] Done! Profiles: {list(self.video_processor.lip_detector.calibration_profiles.keys())}")
            print(f"  [calibration] is_calibrated = {self.video_processor.lip_detector.is_calibrated}")
            return
        
        emotion = self._calibration_queue.pop(0)
        self.video_processor.calibration_emotion = emotion
        self.video_processor.calibration_samples = []
        self.video_processor.calibration_capture = False
        
        prompts = {
            'neutral': "Look relaxed with a neutral expression",
            'happy': "Give a big smile!",
            'sad': "Make a sad face — frown, droop",
            'angry': "Look angry — furrow brows, tense jaw",
            'surprise': "Act surprised — wide eyes, open mouth",
            'fear': "Show fear — wide eyes, tense",
            'disgust': "Show disgust — scrunch nose, squint",
        }
        self.video_processor.calibration_prompt = f"{emotion.upper()}: {prompts.get(emotion, emotion)}"
        
        # 3-second countdown, then capture for 2 seconds
        self.video_processor.calibration_countdown = 3
        self._calibration_tick()
    
    def _calibration_tick(self):
        """Handle calibration countdown and capture timing."""
        if not self.video_processor.calibrating:
            return
        
        countdown = self.video_processor.calibration_countdown
        
        if countdown > 0:
            # Counting down
            self.video_processor.calibration_countdown = countdown - 1
            self.root.after(1000, self._calibration_tick)
        elif not self.video_processor.calibration_capture:
            # Start capturing
            self.video_processor.calibration_capture = True
            self.video_processor.calibration_samples = []
            # Capture for 3 seconds then stop
            self.root.after(3000, self._calibration_finish_emotion)
    
    def _calibration_finish_emotion(self):
        """Finish capturing the current emotion and store calibration data."""
        if not self.video_processor.calibrating:
            return
        
        self.video_processor.calibration_capture = False
        emotion = self.video_processor.calibration_emotion
        samples = self.video_processor.calibration_samples
        
        print(f"  [calibration] {emotion}: collected {len(samples)} samples")
        if samples and self.video_processor.lip_detector is not None:
            self.video_processor.lip_detector.store_calibration(emotion, samples)
            self.status_label.config(text=f"✓ {emotion}: {len(samples)} samples captured")
        else:
            self.status_label.config(text=f"⚠ {emotion}: no samples captured (face not detected?)")
        
        # Move to next emotion after a brief pause
        self.root.after(500, self._calibrate_next_emotion)
    
    def toggle_audio(self):
        """Start or stop audio analysis."""
        if not self.audio_processor.listening:
            self.audio_btn.config(text="🎤 Stop Audio", bg='#e94560')
            if not self.audio_processor.running:
                self.audio_processor.start(
                    self.on_audio_result, 
                    self.on_audio_status
                )
            # Set volume callback for UI updates
            self.audio_processor.volume_callback = self.on_volume_update
            self.audio_processor.start_listening()
            self.audio_status_label.config(text="Audio: Listening...", fg='#0f0')
        else:
            self.audio_btn.config(text="🎤 Start Audio", bg='#16213e')
            self.audio_processor.stop_listening()
            self.audio_status_label.config(text="Audio: Stopped", fg='#888')
            # Reset SER status and volume meter
            self.ser_status_label.config(text="SER: Stopped", fg='#888')
            self._reset_volume_meter()
    
    def update_frame(self):
        """Update video frame."""
        if not self.running or self.cap is None:
            return
        
        start_time = time.time()
        
        ret, frame = self.cap.read()
        if ret:
            # Process frame
            result = self.video_processor.process_frame(frame)
            
            # Record training data if recording
            if self.is_recording_training_data:
                self._record_frame_data(frame, result)
            
            # Draw visualization
            display_frame = self.video_processor.draw_visualization(frame, result)
            
            # Show calibration prompt overlay
            if self.video_processor.calibrating and self.video_processor.calibration_prompt:
                h_frame, w_frame = display_frame.shape[:2]
                # Semi-transparent banner at top
                overlay = display_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w_frame, 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
                
                prompt = self.video_processor.calibration_prompt
                # Main prompt text
                cv2.putText(display_frame, prompt, (20, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                
                # Countdown / status
                if self.video_processor.calibration_capture:
                    n = len(self.video_processor.calibration_samples)
                    cv2.putText(display_frame, f"Capturing... ({n} samples)",
                               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    # Green border while capturing
                    cv2.rectangle(display_frame, (0, 0), (w_frame-1, h_frame-1), (0, 255, 0), 3)
                elif self.video_processor.calibration_countdown > 0:
                    cv2.putText(display_frame, f"Hold expression... {self.video_processor.calibration_countdown}",
                               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Show recording indicator
            if self.is_recording_training_data:
                recording_duration = time.time() - self.recording_start_time
                cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)
                cv2.putText(display_frame, f"REC {recording_duration:.1f}s", 
                           (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Update recording stats in UI
                num_frames = len(self.recorded_frames)
                self.recording_stats_label.config(
                    text=f"Frames: {num_frames} | Duration: {recording_duration:.1f}s"
                )
            
            # Convert to PhotoImage
            display_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(display_rgb)
            img = img.resize((640, 480), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            
            # Update canvas
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.video_canvas.image = photo  # Keep reference
            
            # Update emotion display
            self.update_emotion_display(result)
            
            # Calculate FPS
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            self.fps_counter.append(fps)
            avg_fps = np.mean(self.fps_counter)
            self.fps_label.config(text=f"FPS: {avg_fps:.1f}")
        
        # Schedule next update (30ms = ~33 FPS, reduced from 10ms to prevent CPU overload)
        if self.running:
            self.root.after(30, self.update_frame)
    
    def _record_frame_data(self, frame: np.ndarray, result: dict):
        """Record frame data for training."""
        # Store original frame (resize to 224x224 for model compatibility)
        frame_resized = cv2.resize(frame, (224, 224))
        self.recorded_frames.append(frame_resized)
        
        # Store display frame with mask overlay if checkbox is checked
        display_frame = frame.copy()
        if self.mask_var.get() and result.get('segmentation') is not None and self.video_processor.lip_detector is not None:
            try:
                display_frame = self.video_processor.lip_detector.visualize_segmentation(
                    display_frame, 
                    result['segmentation'],
                    alpha=0.4
                )
            except:
                pass  # Use original if segmentation fails
        self.recorded_display_frames.append(display_frame)
        
        # Store landmarks (468 points x 3 coords)
        if result['landmarks'] is not None:
            self.recorded_landmarks.append(result['landmarks'].copy())
        else:
            # Store zeros if no landmarks detected
            self.recorded_landmarks.append(np.zeros((468, 3), dtype=np.float32))
        
        # Store emotion with timestamp
        self.recorded_emotions.append({
            'frame_idx': len(self.recorded_frames) - 1,
            'timestamp': time.time() - self.recording_start_time,
            'emotion': result['emotion'],
            'confidence': float(result['confidence']),
            'probabilities': result['probabilities'].tolist()
        })
        
        # Store audio emotion result if available
        if hasattr(self, 'ser_emotion') and self.ser_timestamp and \
           (time.time() - self.ser_timestamp) < 0.5:  # Recent audio result
            self.recorded_audio_emotions.append({
                'frame_idx': len(self.recorded_frames) - 1,
                'timestamp': time.time() - self.recording_start_time,
                'emotion': self.ser_emotion,
                'confidence': float(self.ser_confidence),
                'probabilities': self.ser_probs.tolist() if hasattr(self.ser_probs, 'tolist') else list(self.ser_probs)
            })
    
    def update_emotion_display(self, result: dict):
        """Update emotion labels and bars with late fusion of FER (video) and SER (audio)."""
        fer_emotion = result['emotion']
        fer_confidence = result['confidence']
        fer_probs = result['probabilities']
        
        # Check if we have recent SER data (within last 3 seconds)
        ser_age = time.time() - self.ser_timestamp
        use_fusion = ser_age < 3.0 and np.sum(self.ser_probs) > 0
        
        if use_fusion:
            # Late fusion: weighted average of FER and SER probabilities
            # Audio weight = 0.4, Video weight = 0.6 (video is generally more reliable)
            fused_probs = (1 - self.fusion_weight_audio) * fer_probs + self.fusion_weight_audio * self.ser_probs
            
            # Normalize to ensure sum = 1
            fused_probs = fused_probs / np.sum(fused_probs)
            
            # Get dominant emotion from fused probabilities
            emotion_idx = np.argmax(fused_probs)
            emotion = EMOTION_LABELS[emotion_idx]
            confidence = fused_probs[emotion_idx]
            probs = fused_probs
            
            # Update main emotion label with fusion indicator
            color = EMOTION_COLORS.get(emotion, '#808080')
            self.emotion_label.config(text=f"🔗 {emotion.upper()}", fg=color)
            self.confidence_label.config(text=f"Fused: {confidence:.0%} (V:{fer_confidence:.0%} A:{self.ser_confidence:.0%})")
        else:
            # No recent SER data - use FER only
            emotion = fer_emotion
            confidence = fer_confidence
            probs = fer_probs
            
            # Update main emotion label (video only)
            color = EMOTION_COLORS.get(emotion, '#808080')
            self.emotion_label.config(text=emotion.upper(), fg=color)
            self.confidence_label.config(text=f"Confidence: {confidence:.0%}")
        
        # Update probability bars with fused/fer probabilities
        for i, emotion_name in enumerate(EMOTION_LABELS):
            if i < len(probs):
                prob = probs[i]
                canvas, bar = self.prob_bars[emotion_name]
                width = int(prob * 200)  # Scale to canvas width
                canvas.coords(bar, 0, 0, width, 20)
                self.prob_labels[emotion_name].config(text=f"{int(prob*100)}%")
    
    def on_audio_result(self, result: dict):
        """Handle audio result (called from audio thread)."""
        # Use after() to update UI from main thread
        self.root.after(0, lambda: self._update_audio_result(result))
    
    def _update_audio_result(self, result: dict):
        """Update audio result in main thread - SER results stored for fusion."""
        emotion_scores = result.get('emotion_scores', {})
        is_ser = result.get('is_ser', False)
        
        # Handle SER (Speech Emotion Recognition) results - store for fusion
        if is_ser and emotion_scores:
            # Store SER results for fusion with FER
            self.ser_probs = np.array([emotion_scores.get(e, 0.0) for e in EMOTION_LABELS])
            dominant = max(emotion_scores.items(), key=lambda x: x[1])
            self.ser_emotion = dominant[0]
            self.ser_confidence = dominant[1]
            self.ser_timestamp = time.time()
            
            color = EMOTION_COLORS.get(self.ser_emotion, '#0ff')
            
            # Update SER status label (shows raw SER result)
            self.ser_status_label.config(
                text=f"🎤 {self.ser_emotion.upper()} ({self.ser_confidence:.0%})",
                fg=color
            )
            
            # Also update audio status
            self.audio_status_label.config(
                text=f"SER: {self.ser_emotion.upper()} ({self.ser_confidence:.0%})",
                fg='#0ff'
            )
            
            # Update probability bars with SER results immediately
            for i, emotion_name in enumerate(EMOTION_LABELS):
                if i < len(self.ser_probs):
                    prob = self.ser_probs[i]
                    canvas, bar = self.prob_bars[emotion_name]
                    width = int(prob * 200)  # Scale to canvas width
                    canvas.coords(bar, 0, 0, width, 20)
                    self.prob_labels[emotion_name].config(text=f"{int(prob*100)}%")
            
            # NOTE: Don't update probability bars here - fusion happens in update_emotion_display
    
    def on_audio_status(self, status: str):
        """Handle audio status update."""
        self.root.after(0, lambda: self.audio_status_label.config(text=f"Audio: {status}"))
    
    def on_volume_update(self, volume: float, peak: float):
        """Handle volume update from audio thread."""
        # Use after() to update UI from main thread
        self.root.after(0, lambda: self._update_volume_meter(volume, peak))
    
    def _update_volume_meter(self, volume: float, peak: float):
        """Update volume meter in main thread."""
        # Calculate bar width (max 180 pixels)
        bar_width = int(volume * 180)
        peak_x = int(peak * 180)
        
        # Update bar - color based on volume level
        if volume < 0.3:
            color = '#00ff00'  # Green - normal
        elif volume < 0.7:
            color = '#ffff00'  # Yellow - good
        else:
            color = '#ff6600'  # Orange - loud
        
        self.mic_vol_canvas.coords(self.mic_vol_bar, 0, 0, bar_width, 20)
        self.mic_vol_canvas.itemconfig(self.mic_vol_bar, fill=color)
        
        # Update peak indicator
        self.mic_vol_canvas.coords(self.mic_peak_line, peak_x, 0, peak_x, 20)
        
        # Update percentage label
        pct = int(volume * 100)
        self.mic_vol_pct_label.config(text=f"{pct}%", fg=color)
    
    def _reset_volume_meter(self):
        """Reset volume meter to zero."""
        self.mic_vol_canvas.coords(self.mic_vol_bar, 0, 0, 0, 20)
        self.mic_vol_canvas.coords(self.mic_peak_line, 0, 0, 0, 20)
        self.mic_vol_pct_label.config(text="0%", fg='#0f0')

    def toggle_recording(self):
        """Toggle training data recording."""
        if not self.is_recording_training_data:
            self.start_training_recording()
        else:
            self.stop_training_recording()
    
    def start_training_recording(self):
        """Start recording training data (video, audio, landmarks, emotions)."""
        if not self.running:
            self.status_label.config(text="⚠️ Start camera first to record training data!")
            return
        
        self.is_recording_training_data = True
        self.recording_start_time = time.time()
        self.recorded_frames = []
        self.recorded_display_frames = []
        self.recorded_landmarks = []
        self.recorded_emotions = []
        self.recorded_audio_emotions = []
        
        # Start audio recording — ensure audio capture is running
        if not self.audio_processor.running:
            self.audio_processor.start(
                self.on_audio_result,
                self.on_audio_status
            )
            self.audio_processor.volume_callback = self.on_volume_update
        if not self.audio_processor.listening:
            self.audio_processor.start_listening()
            self.audio_btn.config(text="🎤 Stop Audio", bg='#e94560')
            self.audio_status_label.config(text="Audio: Listening...", fg='#0f0')
        self.audio_processor.start_recording()
        
        self.record_btn.config(text="⏹ Stop Recording", bg='#e94560')
        self.recording_status_label.config(text="🔴 RECORDING", fg='#f00')
        self.status_label.config(text="📹 Recording training data... Perform emotions for labeling!")
    
    def stop_training_recording(self):
        """Stop recording and save training data."""
        self.is_recording_training_data = False
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        
        # Stop audio recording and get data
        audio_data = self.audio_processor.stop_recording()
        
        self.record_btn.config(text="⏺ Start Recording", bg='#16213e')
        self.recording_status_label.config(text="Saving...", fg='#ff0')
        self.status_label.config(text="💾 Saving training data...")
        
        # Save in background thread
        save_thread = threading.Thread(
            target=self._save_training_data,
            args=(audio_data, duration),
            daemon=True
        )
        save_thread.start()
    
    def _save_training_data(self, audio_data: np.ndarray, duration: float):
        """Save training data to disk in format compatible with model training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.training_data_dir / f"session_{timestamp}"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        num_frames = len(self.recorded_frames)
        
        try:
            # 1. Save video frames as numpy array (B, T, H, W, C)
            if self.recorded_frames:
                frames_array = np.array(self.recorded_frames)  # (T, H, W, C)
                np.save(session_dir / 'frames.npy', frames_array)
                
                # Also save as individual images for verification
                frames_dir = session_dir / 'frames'
                frames_dir.mkdir(exist_ok=True)
                for i, frame in enumerate(self.recorded_frames):
                    cv2.imwrite(str(frames_dir / f'frame_{i:05d}.jpg'), frame)
            
            # 2. Save landmarks as numpy array (B, T, 468*3)
            if self.recorded_landmarks:
                landmarks_array = np.array(self.recorded_landmarks)  # (T, 468, 3)
                # Flatten to match model input format
                landmarks_flat = landmarks_array.reshape(landmarks_array.shape[0], -1)  # (T, 468*3)
                np.save(session_dir / 'landmarks.npy', landmarks_flat)
            
            # 3. Save audio waveform (for acoustic model)
            if len(audio_data) > 0:
                # Save as numpy (16kHz, normalized float32)
                np.save(session_dir / 'audio_waveform.npy', audio_data)
                
                # Also save as WAV file
                audio_int16 = (audio_data * 32767).astype(np.int16)
                with wave.open(str(session_dir / 'audio.wav'), 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(AudioProcessor.SAMPLE_RATE)
                    wav_file.writeframes(audio_int16.tobytes())
            
            # 4. Save emotion labels with timestamps
            if self.recorded_emotions:
                emotions_data = {
                    'emotions': self.recorded_emotions,
                    'num_frames': num_frames,
                    'duration': duration,
                    'fps': num_frames / duration if duration > 0 else 0
                }
                with open(session_dir / 'emotions.json', 'w') as f:
                    json.dump(emotions_data, f, indent=2)
            
            # 5. Export video with mask overlay (if recorded)
            if self.recorded_display_frames:
                fps = num_frames / duration if duration > 0 else 30
                h, w = self.recorded_display_frames[0].shape[:2]
                # Write raw video with mp4v first (always works)
                raw_path = session_dir / 'recording_raw.mp4'
                video_path = session_dir / 'recording_with_mask.mp4'
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(raw_path), fourcc, fps, (w, h))
                for frame in self.recorded_display_frames:
                    out.write(frame)
                out.release()
                
                # Re-encode with ffmpeg to browser-compatible H.264
                import subprocess
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-i', str(raw_path),
                        '-c:v', 'libx264', '-preset', 'fast',
                        '-pix_fmt', 'yuv420p',  # required for browser playback
                        '-movflags', '+faststart',  # puts moov atom at start for streaming
                        str(video_path)
                    ], capture_output=True, timeout=120)
                    raw_path.unlink()  # remove raw file
                    print(f"✓ Video saved (H.264): {video_path}")
                except Exception as e:
                    # ffmpeg failed — keep raw file as fallback
                    raw_path.rename(video_path)
                    print(f"⚠ ffmpeg re-encode failed ({e}), using raw mp4v")
            
            # 6. Generate audio spectrogram
            spectrogram_path = None
            if len(audio_data) > 0:
                spectrogram_path = self._generate_spectrogram(
                    audio_data, 
                    AudioProcessor.SAMPLE_RATE,
                    session_dir / 'spectrogram.png'
                )
            
            # 7. Save audio emotion results
            if self.recorded_audio_emotions:
                with open(session_dir / 'audio_emotions.json', 'w') as f:
                    json.dump({
                        'audio_emotions': self.recorded_audio_emotions,
                        'num_detections': len(self.recorded_audio_emotions)
                    }, f, indent=2)
            
            # 8. Generate HTML visualization
            html_path = self._generate_html_visualization(
                session_dir,
                timestamp,
                duration,
                num_frames,
                len(audio_data)
            )
            
            # 9. Save metadata for training
            metadata = {
                'timestamp': timestamp,
                'duration_seconds': duration,
                'num_frames': num_frames,
                'audio_sample_rate': AudioProcessor.SAMPLE_RATE,
                'audio_samples': len(audio_data),
                'num_landmarks_frames': len(self.recorded_landmarks),
                'num_emotion_labels': len(self.recorded_emotions),
                'num_audio_emotion_labels': len(self.recorded_audio_emotions),
                'video_file': 'recording_with_mask.mp4',
                'spectrogram_file': 'spectrogram.png' if spectrogram_path else None,
                'html_visualization': 'visualization.html',
                'format_info': {
                    'frames': '(T, H, W, C) - BGR uint8',
                    'landmarks': '(T, 1404) - MediaPipe 468 landmarks x 3 coords',
                    'audio_waveform': '(samples,) - 16kHz float32 normalized [-1,1]',
                    'emotions': 'list of {frame_idx, emotion, confidence, probabilities}'
                }
            }
            with open(session_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Update UI from main thread
            self.root.after(0, lambda: self._recording_saved(session_dir, num_frames, duration))
            
        except Exception as e:
            print(f"Error saving training data: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.status_label.config(text=f"❌ Save error: {str(e)[:40]}"))
    
    def _recording_saved(self, session_dir: Path, num_frames: int, duration: float):
        """Called when recording is saved successfully."""
        self.recording_status_label.config(text="Not Recording", fg='#888')
        self.status_label.config(text=f"✅ Saved to {session_dir.name} ({num_frames} frames, {duration:.1f}s) - Open visualization.html in browser")
        self.recording_stats_label.config(text=f"Last: {num_frames} frames | {duration:.1f}s")
    
    def _generate_spectrogram(self, audio_data: np.ndarray, sample_rate: int, output_path: Path) -> Path:
        """Generate and save audio spectrogram."""
        try:
            # Compute spectrogram
            f, t, Sxx = scipy_signal.spectrogram(audio_data, fs=sample_rate, nperseg=1024)
            
            # Create figure
            plt.figure(figsize=(12, 4))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.title('Audio Spectrogram')
            plt.colorbar(label='Power [dB]')
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Spectrogram saved: {output_path}")
            return output_path
        except Exception as e:
            print(f"Warning: Failed to generate spectrogram: {e}")
            return None
    
    def _generate_html_visualization(self, session_dir: Path, timestamp: str, 
                                     duration: float, num_frames: int, 
                                     num_audio_samples: int) -> Path:
        """Generate HTML file for visualizing recording results."""
        try:
            html_path = session_dir / 'visualization.html'
            
            # Prepare time-series data for emotions
            video_emotions_ts = []
            audio_emotions_ts = []
            
            if self.recorded_emotions:
                video_emotions_ts = [
                    {'time': e['timestamp'], 'emotion': e['emotion'], 
                     'confidence': e['confidence']} 
                    for e in self.recorded_emotions
                ]
            
            if self.recorded_audio_emotions:
                audio_emotions_ts = [
                    {'time': e['timestamp'], 'emotion': e['emotion'], 
                     'confidence': e['confidence']} 
                    for e in self.recorded_audio_emotions
                ]
            
            # Create HTML content
            html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Emotion Recording - {timestamp}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            text-align: center;
            color: #e94560;
            margin-bottom: 10px;
        }}
        .subtitle {{
            text-align: center;
            color: #aaa;
            margin-bottom: 30px;
        }}
        .info-bar {{
            background: #0f3460;
            padding: 15px;
            border-radius: 8px;
            display: flex;
            justify-content: space-around;
            margin-bottom: 30px;
        }}
        .info-item {{
            text-align: center;
        }}
        .info-label {{
            color: #888;
            font-size: 12px;
            text-transform: uppercase;
        }}
        .info-value {{
            color: #0ff;
            font-size: 24px;
            font-weight: bold;
        }}
        .section {{
            background: rgba(15, 52, 96, 0.4);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            border: 1px solid #0f3460;
        }}
        h2 {{
            color: #e94560;
            border-bottom: 2px solid #0f3460;
            padding-bottom: 10px;
        }}
        video {{
            width: 100%;
            max-width: 800px;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
        }}
        img.spectrogram {{
            width: 100%;
            max-width: 1200px;
            border-radius: 8px;
            display: block;
            margin: 20px auto;
            background: white;
            padding: 10px;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        .emotion-colors {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }}
        .color-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .color-box {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎭 Multimodal Emotion Recording</h1>
        <div class="subtitle">Session: {timestamp}</div>
        
        <div class="info-bar">
            <div class="info-item">
                <div class="info-label">Duration</div>
                <div class="info-value">{duration:.1f}s</div>
            </div>
            <div class="info-item">
                <div class="info-label">Frames</div>
                <div class="info-value">{num_frames}</div>
            </div>
            <div class="info-item">
                <div class="info-label">FPS</div>
                <div class="info-value">{(num_frames/duration if duration > 0 else 0):.1f}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Audio Samples</div>
                <div class="info-value">{num_audio_samples}</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📹 Video Recording (with Mask Overlay)</h2>
            <video controls>
                <source src="recording_with_mask.mp4" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div class="section">
            <h2>🎵 Audio Spectrogram</h2>
            <img src="spectrogram.png" alt="Audio Spectrogram" class="spectrogram">
        </div>
        
        <div class="section">
            <h2>📊 Video Emotion Time Series</h2>
            <div class="chart-container">
                <canvas id="videoEmotionsChart"></canvas>
            </div>
        </div>
        
        <div class="section">
            <h2>📊 Audio Emotion Time Series</h2>
            <div class="chart-container">
                <canvas id="audioEmotionsChart"></canvas>
            </div>
        </div>
        
        <div class="emotion-colors">
            <div class="color-item"><div class="color-box" style="background:#FF0000"></div><span>Angry</span></div>
            <div class="color-item"><div class="color-box" style="background:#808000"></div><span>Disgust</span></div>
            <div class="color-item"><div class="color-box" style="background:#FF00FF"></div><span>Fear</span></div>
            <div class="color-item"><div class="color-box" style="background:#00FF00"></div><span>Happy</span></div>
            <div class="color-item"><div class="color-box" style="background:#0000FF"></div><span>Sad</span></div>
            <div class="color-item"><div class="color-box" style="background:#FFFF00"></div><span>Surprise</span></div>
            <div class="color-item"><div class="color-box" style="background:#808080"></div><span>Neutral</span></div>
        </div>
    </div>
    
    <script>
        // Emotion color mapping
        const emotionColors = {{
            'angry': 'rgba(255, 0, 0, 0.8)',
            'disgust': 'rgba(128, 128, 0, 0.8)',
            'fear': 'rgba(255, 0, 255, 0.8)',
            'happy': 'rgba(0, 255, 0, 0.8)',
            'sad': 'rgba(0, 0, 255, 0.8)',
            'surprise': 'rgba(255, 255, 0, 0.8)',
            'neutral': 'rgba(128, 128, 128, 0.8)'
        }};
        
        // Video emotions data
        const videoEmotionsData = {json.dumps(video_emotions_ts)};
        
        // Audio emotions data
        const audioEmotionsData = {json.dumps(audio_emotions_ts)};
        
        // Create video emotions chart
        const videoCtx = document.getElementById('videoEmotionsChart').getContext('2d');
        new Chart(videoCtx, {{
            type: 'scatter',
            data: {{
                datasets: Object.keys(emotionColors).map(emotion => {{
                    return {{
                        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                        data: videoEmotionsData
                            .filter(d => d.emotion === emotion)
                            .map(d => {{ return {{x: d.time, y: d.confidence}}; }}),
                        backgroundColor: emotionColors[emotion],
                        borderColor: emotionColors[emotion],
                        pointRadius: 4
                    }};
                }})
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'linear',
                        title: {{display: true, text: 'Time (seconds)', color: '#eee'}},
                        ticks: {{color: '#aaa'}},
                        grid: {{color: 'rgba(255,255,255,0.1)'}}
                    }},
                    y: {{
                        title: {{display: true, text: 'Confidence', color: '#eee'}},
                        min: 0,
                        max: 1,
                        ticks: {{color: '#aaa'}},
                        grid: {{color: 'rgba(255,255,255,0.1)'}}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{color: '#eee'}}
                    }},
                    title: {{
                        display: true,
                        text: 'Video-based Emotion Detection Over Time',
                        color: '#e94560',
                        font: {{size: 16}}
                    }}
                }}
            }}
        }});
        
        // Create audio emotions chart
        const audioCtx = document.getElementById('audioEmotionsChart').getContext('2d');
        new Chart(audioCtx, {{
            type: 'scatter',
            data: {{
                datasets: Object.keys(emotionColors).map(emotion => {{
                    return {{
                        label: emotion.charAt(0).toUpperCase() + emotion.slice(1),
                        data: audioEmotionsData
                            .filter(d => d.emotion === emotion)
                            .map(d => {{ return {{x: d.time, y: d.confidence}}; }}),
                        backgroundColor: emotionColors[emotion],
                        borderColor: emotionColors[emotion],
                        pointRadius: 6
                    }};
                }})
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        type: 'linear',
                        title: {{display: true, text: 'Time (seconds)', color: '#eee'}},
                        ticks: {{color: '#aaa'}},
                        grid: {{color: 'rgba(255,255,255,0.1)'}}
                    }},
                    y: {{
                        title: {{display: true, text: 'Confidence', color: '#eee'}},
                        min: 0,
                        max: 1,
                        ticks: {{color: '#aaa'}},
                        grid: {{color: 'rgba(255,255,255,0.1)'}}
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{color: '#eee'}}
                    }},
                    title: {{
                        display: true,
                        text: 'Audio-based Emotion Detection Over Time',
                        color: '#e94560',
                        font: {{size: 16}}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            print(f"✓ HTML visualization saved: {html_path}")
            return html_path
            
        except Exception as e:
            print(f"Warning: Failed to generate HTML visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def on_close(self):
        """Handle window close."""
        # Stop recording if active
        if self.is_recording_training_data:
            self.stop_training_recording()
        
        self.running = False
        if self.cap:
            self.cap.release()
        self.audio_processor.stop()
        self.root.destroy()
    
    def run(self):
        """Run the application."""
        self.root.mainloop()


def main():
    """Main entry point."""
    app = EmotionDemoApp()
    app.run()


if __name__ == "__main__":
    main()

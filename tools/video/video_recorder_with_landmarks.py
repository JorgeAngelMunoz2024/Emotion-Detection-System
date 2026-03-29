"""
Video Recorder with Facial Landmarks and Attention Focus
Records video with MediaPipe facial landmarks overlay and attention heatmaps.
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
import time
from datetime import datetime
from typing import Optional, List, Tuple
import argparse
import os

from models.emotion_detector import HybridEmotionRecognizer, EMOTION_LABELS
from models.mediapipe_detector import MediaPipeEmotionDetector


class VideoRecorderWithLandmarks:
    """
    Records video with facial landmarks, attention maps, and emotion predictions.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mediapipe_model: bool = True,
        sequence_length: int = 16,
        device: str = 'cpu',
        output_dir: str = 'recordings',
        show_landmarks: bool = True,
        show_attention: bool = True,
        fps: int = 30
    ):
        """
        Args:
            model_path: Path to pretrained model checkpoint
            use_mediapipe_model: Use MediaPipe-enhanced model
            sequence_length: Number of frames for temporal modeling
            device: 'cpu' or 'cuda'
            output_dir: Directory to save recordings
            show_landmarks: Overlay facial landmarks on video
            show_attention: Show attention heatmap overlay
            fps: Frames per second for recording
        """
        self.device = torch.device(device)
        self.sequence_length = sequence_length
        self.output_dir = output_dir
        self.show_landmarks = show_landmarks
        self.show_attention = show_attention
        self.target_fps = fps
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Load emotion detection model
        if use_mediapipe_model:
            self.model = MediaPipeEmotionDetector(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=True,
                fusion_method='concat'
            ).to(self.device)
        else:
            self.model = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=True,
                temporal_layers=2,
                temporal_heads=4,
                fusion_method='weighted'
            ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded model from {model_path}")
        
        self.model.eval()
        self.use_mediapipe_model = use_mediapipe_model
        
        # Frame buffer for temporal modeling
        self.frame_buffer = deque(maxlen=sequence_length)
        self.landmark_buffer = deque(maxlen=sequence_length)
        
        # Face detector (backup for when MediaPipe fails)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion colors (BGR format)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 128),  # Dark Yellow
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 255), # Yellow
            'neutral': (128, 128, 128) # Gray
        }
        
        # Key facial landmark regions for attention focus
        self.landmark_regions = {
            'left_eye': list(range(133, 155)),
            'right_eye': list(range(362, 384)),
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
            'nose': [1, 2, 98, 327]
        }
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.recording_start_time = None
        self.frame_count = 0
        
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks using MediaPipe.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Landmarks array (468, 3) or None if no face detected
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            # Convert normalized coordinates to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z  # Relative depth
                landmark_points.append([x, y, z])
            
            return np.array(landmark_points, dtype=np.float32)
        
        return None
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray, 
                      highlight_regions: Optional[List[str]] = None):
        """
        Draw facial landmarks on frame with optional region highlighting.
        
        Args:
            frame: BGR image frame
            landmarks: (468, 3) array of landmarks
            highlight_regions: List of region names to highlight
        """
        h, w = frame.shape[:2]
        
        # Draw all landmarks as small circles
        for i, (x, y, z) in enumerate(landmarks):
            color = (0, 255, 0)  # Default green
            radius = 1
            
            # Highlight specific regions
            if highlight_regions:
                for region_name in highlight_regions:
                    if i in self.landmark_regions.get(region_name, []):
                        color = (0, 0, 255)  # Red for highlighted
                        radius = 2
                        break
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        
        # Draw connections for key regions
        if highlight_regions:
            for region_name in highlight_regions:
                indices = self.landmark_regions.get(region_name, [])
                if len(indices) > 1:
                    points = landmarks[indices][:, :2].astype(np.int32)
                    cv2.polylines(frame, [points], isClosed=True, 
                                color=(255, 0, 0), thickness=2)
    
    def get_attention_weights(self, face_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Extract attention weights from the model.
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            Attention map as numpy array or None
        """
        try:
            with torch.no_grad():
                # Access attention module from spatial CNN
                if hasattr(self.model, 'spatial_cnn') and hasattr(self.model.spatial_cnn, 'spatial_attention'):
                    # Forward pass through CNN to get features
                    features = self.model.spatial_cnn.backbone(face_tensor)
                    
                    # Get attention map
                    attention = self.model.spatial_cnn.spatial_attention(features)
                    attention_map = attention[0, 0].cpu().numpy()
                    
                    return attention_map
                elif hasattr(self.model, 'cnn') and hasattr(self.model.cnn, 'spatial_attention'):
                    features = self.model.cnn.backbone(face_tensor)
                    attention = self.model.cnn.spatial_attention(features)
                    attention_map = attention[0, 0].cpu().numpy()
                    
                    return attention_map
        except Exception as e:
            print(f"Warning: Could not extract attention weights: {e}")
        
        return None
    
    def overlay_attention_heatmap(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                 attention_map: np.ndarray):
        """
        Overlay attention heatmap on face region.
        
        Args:
            frame: BGR image frame
            face_bbox: (x, y, w, h) face bounding box
            attention_map: Attention weights array
        """
        x, y, w, h = face_bbox
        
        # Resize attention map to face size
        attention_resized = cv2.resize(attention_map, (w, h))
        
        # Normalize to 0-255
        attention_norm = ((attention_resized - attention_resized.min()) / 
                         (attention_resized.max() - attention_resized.min() + 1e-8) * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(attention_norm, cv2.COLORMAP_JET)
        
        # Blend with original frame
        face_region = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 0.6, heatmap, 0.4, 0)
        frame[y:y+h, x:x+w] = blended
    
    def preprocess_frame(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1))
        face_tensor = face_tensor.unsqueeze(0)
        
        return face_tensor.to(self.device)
    
    def detect_emotion(self, face_tensor: torch.Tensor, 
                      landmarks: Optional[np.ndarray] = None) -> Tuple[str, float, np.ndarray]:
        """
        Detect emotion from face tensor and optional landmarks.
        
        Returns:
            (emotion_label, confidence, probabilities)
        """
        with torch.no_grad():
            if self.use_mediapipe_model and landmarks is not None:
                # Use MediaPipe-enhanced model
                landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
                logits = self.model(face_tensor, landmarks_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            else:
                # Use standard model
                logits = self.model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]
            emotion_label = EMOTION_LABELS[emotion_idx]
            
            return emotion_label, confidence, probs
    
    def start_recording(self, frame_width: int, frame_height: int):
        """Start video recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f"recording_{timestamp}.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            output_path, fourcc, self.target_fps, (frame_width, frame_height)
        )
        
        self.is_recording = True
        self.recording_start_time = time.time()
        self.frame_count = 0
        
        print(f"✓ Started recording: {output_path}")
        return output_path
    
    def stop_recording(self):
        """Stop video recording."""
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        self.is_recording = False
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        
        print(f"✓ Recording stopped. Duration: {duration:.1f}s, Frames: {self.frame_count}")
    
    def draw_ui(self, frame: np.ndarray, emotion: str, confidence: float, 
                probs: np.ndarray, fps: float, face_bbox: Optional[Tuple] = None):
        """Draw UI overlays on frame."""
        h, w = frame.shape[:2]
        
        # Draw recording indicator
        if self.is_recording:
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            duration = time.time() - self.recording_start_time
            time_text = f"REC {duration:.1f}s"
            cv2.putText(frame, time_text, (w - 120, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw emotion if face detected
        if face_bbox:
            x, y, w_box, h_box = face_bbox
            color = self.emotion_colors[emotion]
            
            # Face bounding box
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            # Emotion label
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw emotion probability bars
        bar_height = 20
        bar_width = 150
        start_x = w - bar_width - 20
        start_y = 60
        
        for i, (emotion_label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
            y_pos = start_y + i * (bar_height + 5)
            
            # Background
            cv2.rectangle(frame, (start_x, y_pos), 
                         (start_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Probability bar
            prob_width = int(bar_width * prob)
            color = self.emotion_colors[emotion_label]
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + prob_width, y_pos + bar_height),
                         color, -1)
            
            # Label
            text = f"{emotion_label}: {prob:.2f}"
            cv2.putText(frame, text, (start_x + 5, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw controls
        controls = [
            "Controls:",
            "R - Start/Stop Recording",
            "L - Toggle Landmarks",
            "A - Toggle Attention",
            "S - Save Screenshot",
            "Q - Quit"
        ]
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (10, h - 20 - (len(controls) - i - 1) * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def run(self):
        """Run the video recorder with webcam."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print("=" * 70)
        print("Video Recorder with Facial Landmarks and Attention Focus")
        print("=" * 70)
        print(f"Resolution: {frame_width}x{frame_height} @ {self.target_fps} FPS")
        print(f"Output directory: {self.output_dir}")
        print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
        print(f"Attention: {'ON' if self.show_attention else 'OFF'}")
        print("\nPress 'R' to start/stop recording")
        print("=" * 70)
        
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                display_frame = frame.copy()
                
                # Extract facial landmarks
                landmarks = self.extract_landmarks(frame)
                
                emotion = 'neutral'
                confidence = 0.0
                probs = np.zeros(7)
                face_bbox = None
                
                if landmarks is not None:
                    # Calculate face bounding box from landmarks
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)
                    
                    face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    
                    # Extract face for emotion detection
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        face_tensor = self.preprocess_frame(face_img)
                        
                        # Detect emotion
                        emotion, confidence, probs = self.detect_emotion(face_tensor, landmarks)
                        
                        # Draw attention heatmap
                        if self.show_attention:
                            attention_map = self.get_attention_weights(face_tensor)
                            if attention_map is not None:
                                self.overlay_attention_heatmap(display_frame, face_bbox, attention_map)
                        
                        # Draw landmarks with region highlighting based on emotion
                        if self.show_landmarks:
                            # Highlight different regions based on emotion
                            highlight = []
                            if emotion in ['happy', 'sad', 'disgust']:
                                highlight.append('mouth_outer')
                            if emotion in ['surprise', 'fear', 'angry']:
                                highlight.extend(['left_eyebrow', 'right_eyebrow'])
                            if emotion in ['surprise', 'fear']:
                                highlight.extend(['left_eye', 'right_eye'])
                            
                            self.draw_landmarks(display_frame, landmarks, highlight)
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                
                # Draw UI
                self.draw_ui(display_frame, emotion, confidence, probs, avg_fps, face_bbox)
                
                # Write frame if recording
                if self.is_recording and self.video_writer:
                    self.video_writer.write(display_frame)
                    self.frame_count += 1
                
                # Display
                cv2.imshow('Video Recorder with Landmarks', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.is_recording:
                        self.stop_recording()
                    else:
                        self.start_recording(frame_width, frame_height)
                elif key == ord('l'):
                    self.show_landmarks = not self.show_landmarks
                    print(f"Landmarks: {'ON' if self.show_landmarks else 'OFF'}")
                elif key == ord('a'):
                    self.show_attention = not self.show_attention
                    print(f"Attention: {'ON' if self.show_attention else 'OFF'}")
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_dir, f"screenshot_{timestamp}.jpg")
                    cv2.imwrite(filename, display_frame)
                    print(f"✓ Saved screenshot: {filename}")
        
        finally:
            if self.is_recording:
                self.stop_recording()
            
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            print("\n✓ Video recorder stopped")


def main():
    parser = argparse.ArgumentParser(
        description='Video Recorder with Facial Landmarks and Attention Focus'
    )
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--use-mediapipe', action='store_true',
                       help='Use MediaPipe-enhanced emotion detection model')
    parser.add_argument('--output-dir', type=str, default='recordings',
                       help='Directory to save recordings')
    parser.add_argument('--no-landmarks', action='store_true',
                       help='Disable landmark overlay')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention heatmap overlay')
    parser.add_argument('--fps', type=int, default=30,
                       help='Target FPS for recording')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    
    args = parser.parse_args()
    
    recorder = VideoRecorderWithLandmarks(
        model_path=args.model_path,
        use_mediapipe_model=args.use_mediapipe,
        output_dir=args.output_dir,
        show_landmarks=not args.no_landmarks,
        show_attention=not args.no_attention,
        fps=args.fps,
        device=args.device
    )
    
    recorder.run()


if __name__ == "__main__":
    main()

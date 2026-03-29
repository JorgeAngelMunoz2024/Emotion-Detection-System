"""
Real-time Webcam Emotion Detection
Captures video from webcam and detects emotions in real-time using the hybrid model.
"""

import cv2
import torch
import numpy as np
from collections import deque
import time
from typing import Optional
import argparse

from models.emotion_detector import HybridEmotionRecognizer, EMOTION_LABELS, SpatialAttentionCNN


class WebcamEmotionDetector:
    """
    Real-time emotion detection from webcam feed.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_hybrid: bool = True,
        sequence_length: int = 16,
        device: str = 'cpu',
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to pretrained model checkpoint
            use_hybrid: Use hybrid model (True) or spatial CNN only (False)
            sequence_length: Number of frames for temporal modeling
            device: 'cpu' or 'cuda'
            confidence_threshold: Minimum confidence for emotion display
        """
        self.device = torch.device(device)
        self.use_hybrid = use_hybrid
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Load model
        if use_hybrid:
            self.model = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=True,
                temporal_layers=2,
                temporal_heads=4,
                fusion_method='weighted'
            ).to(self.device)
        else:
            self.model = SpatialAttentionCNN(
                num_emotions=7,
                backbone='mobilenet_v2',
                pretrained=True,
                use_attention=True
            ).to(self.device)
        
        # Load checkpoint if provided
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        
        # Frame buffer for temporal modeling
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion colors (BGR format for OpenCV)
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 128),  # Dark Yellow
            'fear': (255, 0, 255),     # Magenta
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 255), # Yellow
            'neutral': (128, 128, 128) # Gray
        }
        
        # FPS counter
        self.fps_counter = deque(maxlen=30)
        
    def preprocess_frame(self, face_img: np.ndarray) -> torch.Tensor:
        """
        Preprocess face image for model input.
        
        Args:
            face_img: Face image (H, W, 3) in BGR format
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Resize to 224x224
        face_resized = cv2.resize(face_rgb, (224, 224))
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor (C, H, W)
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1))
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor.to(self.device)
    
    def detect_emotion(self, face_tensor: torch.Tensor) -> tuple:
        """
        Detect emotion from face tensor.
        
        Returns:
            (emotion_label, confidence, all_probabilities)
        """
        with torch.no_grad():
            if self.use_hybrid and len(self.frame_buffer) >= self.sequence_length:
                # Use hybrid model with temporal sequence
                frame_sequence = torch.stack(list(self.frame_buffer))
                frame_sequence = frame_sequence.unsqueeze(0)  # Add batch dimension
                
                predictions = self.model.predict_emotion(frame_sequence)
                probs = predictions['combined_probabilities'][0].cpu().numpy()
            else:
                # Use spatial CNN only
                if self.use_hybrid:
                    logits = self.model.spatial_cnn(face_tensor)
                else:
                    logits = self.model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]
            emotion_label = EMOTION_LABELS[emotion_idx]
            
            return emotion_label, confidence, probs
    
    def draw_results(
        self, 
        frame: np.ndarray, 
        x: int, y: int, w: int, h: int,
        emotion: str, 
        confidence: float, 
        probs: np.ndarray,
        fps: float
    ):
        """
        Draw detection results on frame.
        """
        # Draw face bounding box
        color = self.emotion_colors[emotion]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw emotion label
        label = f"{emotion}: {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Background for text
        cv2.rectangle(
            frame, 
            (x, y - label_size[1] - 10), 
            (x + label_size[0], y),
            color, 
            -1
        )
        cv2.putText(
            frame, 
            label, 
            (x, y - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.6, 
            (255, 255, 255), 
            2
        )
        
        # Draw probability bars
        bar_height = 20
        bar_width = 200
        start_y = 30
        
        for i, (emotion_label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
            y_pos = start_y + i * (bar_height + 5)
            
            # Background bar
            cv2.rectangle(
                frame, 
                (10, y_pos), 
                (10 + bar_width, y_pos + bar_height),
                (50, 50, 50), 
                -1
            )
            
            # Probability bar
            prob_width = int(bar_width * prob)
            color = self.emotion_colors[emotion_label]
            cv2.rectangle(
                frame, 
                (10, y_pos), 
                (10 + prob_width, y_pos + bar_height),
                color, 
                -1
            )
            
            # Label
            text = f"{emotion_label}: {prob:.2f}"
            cv2.putText(
                frame, 
                text, 
                (15, y_pos + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.4, 
                (255, 255, 255), 
                1
            )
        
        # Draw FPS and mode
        mode_text = "Hybrid Mode" if self.use_hybrid else "Spatial Mode"
        buffer_text = f"Buffer: {len(self.frame_buffer)}/{self.sequence_length}"
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, mode_text, (frame.shape[1] - 150, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, buffer_text, (frame.shape[1] - 150, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run(self):
        """
        Start webcam emotion detection.
        """
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Starting emotion detection...")
        print(f"Mode: {'Hybrid (CNN + Transformer)' if self.use_hybrid else 'Spatial CNN only'}")
        print("Press 'q' to quit, 's' to save screenshot")
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame")
                    break
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
                )
                
                # Process each face
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Preprocess
                    face_tensor = self.preprocess_frame(face_img)
                    
                    # Add to buffer for temporal modeling
                    self.frame_buffer.append(face_tensor.squeeze(0))
                    
                    # Detect emotion
                    emotion, confidence, probs = self.detect_emotion(face_tensor)
                    
                    # Calculate FPS
                    fps = 1.0 / (time.time() - start_time) if len(self.fps_counter) > 0 else 0
                    self.fps_counter.append(fps)
                    avg_fps = np.mean(self.fps_counter)
                    
                    # Draw results
                    if confidence >= self.confidence_threshold:
                        self.draw_results(frame, x, y, w, h, emotion, confidence, probs, avg_fps)
                
                # Display frame
                cv2.imshow('Real-time Emotion Detection', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"emotion_capture_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved screenshot: {filename}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Emotion detection stopped")


def main():
    parser = argparse.ArgumentParser(description='Real-time Webcam Emotion Detection')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--spatial-only', action='store_true',
                        help='Use spatial CNN only (no temporal modeling)')
    parser.add_argument('--sequence-length', type=int, default=16,
                        help='Number of frames for temporal sequence')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                        help='Device to use for inference')
    parser.add_argument('--confidence', type=float, default=0.3,
                        help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    # Create detector
    detector = WebcamEmotionDetector(
        model_path=args.model_path,
        use_hybrid=not args.spatial_only,
        sequence_length=args.sequence_length,
        device=args.device,
        confidence_threshold=args.confidence
    )
    
    # Run detection
    detector.run()


if __name__ == "__main__":
    main()

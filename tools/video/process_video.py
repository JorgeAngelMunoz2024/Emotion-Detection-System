"""
Process Pre-recorded Video with Facial Landmarks and Attention
Analyzes existing video files with emotion detection, landmarks, and attention focus.
"""

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
import time
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import argparse
import os
import json

from models.emotion_detector import HybridEmotionRecognizer, EMOTION_LABELS
from models.mediapipe_detector import MediaPipeEmotionDetector


class VideoProcessor:
    """
    Process video files with facial landmark detection and emotion analysis.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mediapipe_model: bool = True,
        device: str = 'cpu',
        show_landmarks: bool = True,
        show_attention: bool = True,
        save_analysis: bool = True
    ):
        """
        Args:
            model_path: Path to pretrained model checkpoint
            use_mediapipe_model: Use MediaPipe-enhanced model
            device: 'cpu' or 'cuda'
            show_landmarks: Overlay facial landmarks
            show_attention: Show attention heatmap
            save_analysis: Save frame-by-frame analysis to JSON
        """
        self.device = torch.device(device)
        self.show_landmarks = show_landmarks
        self.show_attention = show_attention
        self.save_analysis = save_analysis
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
        
        # Emotion colors
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 128),
            'fear': (255, 0, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (0, 255, 255),
            'neutral': (128, 128, 128)
        }
        
        # Key facial landmark regions
        self.landmark_regions = {
            'left_eye': list(range(133, 155)),
            'right_eye': list(range(362, 384)),
            'left_eyebrow': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
            'right_eyebrow': [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
            'mouth_outer': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
            'nose': [1, 2, 98, 327]
        }
        
        # Analysis data
        self.frame_analysis = []
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial landmarks using MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            
            landmark_points = []
            for landmark in landmarks.landmark:
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z
                landmark_points.append([x, y, z])
            
            return np.array(landmark_points, dtype=np.float32)
        
        return None
    
    def get_attention_focus_regions(self, landmarks: np.ndarray, 
                                   attention_map: np.ndarray) -> Dict[str, float]:
        """
        Calculate attention weights for each facial region.
        
        Args:
            landmarks: (468, 3) facial landmarks
            attention_map: Attention heatmap
            
        Returns:
            Dictionary mapping region names to attention scores
        """
        h, w = attention_map.shape
        attention_scores = {}
        
        for region_name, indices in self.landmark_regions.items():
            if not indices:
                continue
            
            # Get landmark coordinates for this region
            region_landmarks = landmarks[indices][:, :2]
            
            # Normalize to attention map coordinates
            x_coords = (region_landmarks[:, 0] / 224 * w).astype(int)
            y_coords = (region_landmarks[:, 1] / 224 * h).astype(int)
            
            # Clip to valid range
            x_coords = np.clip(x_coords, 0, w - 1)
            y_coords = np.clip(y_coords, 0, h - 1)
            
            # Sample attention values at landmark positions
            attention_values = attention_map[y_coords, x_coords]
            
            # Average attention for this region
            attention_scores[region_name] = float(np.mean(attention_values))
        
        return attention_scores
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray,
                      highlight_regions: Optional[List[str]] = None):
        """Draw facial landmarks with optional region highlighting."""
        for i, (x, y, z) in enumerate(landmarks):
            color = (0, 255, 0)
            radius = 1
            
            if highlight_regions:
                for region_name in highlight_regions:
                    if i in self.landmark_regions.get(region_name, []):
                        color = (0, 0, 255)
                        radius = 2
                        break
            
            cv2.circle(frame, (int(x), int(y)), radius, color, -1)
        
        if highlight_regions:
            for region_name in highlight_regions:
                indices = self.landmark_regions.get(region_name, [])
                if len(indices) > 1:
                    points = landmarks[indices][:, :2].astype(np.int32)
                    cv2.polylines(frame, [points], isClosed=True,
                                color=(255, 0, 0), thickness=2)
    
    def overlay_attention_heatmap(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int],
                                 attention_map: np.ndarray):
        """Overlay attention heatmap on face region."""
        x, y, w, h = face_bbox
        
        attention_resized = cv2.resize(attention_map, (w, h))
        attention_norm = ((attention_resized - attention_resized.min()) /
                         (attention_resized.max() - attention_resized.min() + 1e-8) * 255).astype(np.uint8)
        
        heatmap = cv2.applyColorMap(attention_norm, cv2.COLORMAP_JET)
        face_region = frame[y:y+h, x:x+w]
        blended = cv2.addWeighted(face_region, 0.6, heatmap, 0.4, 0)
        frame[y:y+h, x:x+w] = blended
    
    def get_attention_weights(self, face_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """Extract attention weights from model."""
        try:
            with torch.no_grad():
                if hasattr(self.model, 'spatial_cnn') and hasattr(self.model.spatial_cnn, 'spatial_attention'):
                    features = self.model.spatial_cnn.backbone(face_tensor)
                    attention = self.model.spatial_cnn.spatial_attention(features)
                    return attention[0, 0].cpu().numpy()
                elif hasattr(self.model, 'cnn') and hasattr(self.model.cnn, 'spatial_attention'):
                    features = self.model.cnn.backbone(face_tensor)
                    attention = self.model.cnn.spatial_attention(features)
                    return attention[0, 0].cpu().numpy()
        except Exception as e:
            print(f"Warning: Could not extract attention: {e}")
        
        return None
    
    def preprocess_frame(self, face_img: np.ndarray) -> torch.Tensor:
        """Preprocess face image for model input."""
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        face_tensor = torch.from_numpy(face_normalized.transpose(2, 0, 1))
        return face_tensor.unsqueeze(0).to(self.device)
    
    def detect_emotion(self, face_tensor: torch.Tensor,
                      landmarks: Optional[np.ndarray] = None) -> Tuple[str, float, np.ndarray]:
        """Detect emotion from face tensor and landmarks."""
        with torch.no_grad():
            if self.use_mediapipe_model and landmarks is not None:
                landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
                logits = self.model(face_tensor, landmarks_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            else:
                logits = self.model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]
            emotion_label = EMOTION_LABELS[emotion_idx]
            
            return emotion_label, confidence, probs
    
    def draw_ui(self, frame: np.ndarray, emotion: str, confidence: float,
                probs: np.ndarray, frame_num: int, total_frames: int,
                face_bbox: Optional[Tuple] = None):
        """Draw UI overlays."""
        h, w = frame.shape[:2]
        
        # Progress
        progress = frame_num / total_frames if total_frames > 0 else 0
        cv2.putText(frame, f"Frame {frame_num}/{total_frames} ({progress*100:.1f}%)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw progress bar
        bar_width = w - 40
        bar_height = 10
        cv2.rectangle(frame, (20, 45), (20 + bar_width, 45 + bar_height),
                     (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 45), (20 + int(bar_width * progress), 45 + bar_height),
                     (0, 255, 0), -1)
        
        # Emotion if face detected
        if face_bbox:
            x, y, w_box, h_box = face_bbox
            color = self.emotion_colors[emotion]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            label = f"{emotion}: {confidence:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Emotion bars
        bar_height = 20
        bar_width = 150
        start_x = w - bar_width - 20
        start_y = 70
        
        for i, (emotion_label, prob) in enumerate(zip(EMOTION_LABELS, probs)):
            y_pos = start_y + i * (bar_height + 5)
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            prob_width = int(bar_width * prob)
            color = self.emotion_colors[emotion_label]
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + prob_width, y_pos + bar_height),
                         color, -1)
            
            text = f"{emotion_label}: {prob:.2f}"
            cv2.putText(frame, text, (start_x + 5, y_pos + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def process_video(self, input_path: str, output_path: Optional[str] = None,
                     display: bool = True) -> Dict:
        """
        Process video file with emotion detection and landmark analysis.
        
        Args:
            input_path: Path to input video
            output_path: Path to save processed video (optional)
            display: Show processing in real-time
            
        Returns:
            Dictionary with analysis summary
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{'='*70}")
        print(f"Processing Video: {os.path.basename(input_path)}")
        print(f"{'='*70}")
        print(f"Resolution: {frame_width}x{frame_height}")
        print(f"FPS: {fps:.1f}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.1f}s")
        print(f"{'='*70}\n")
        
        # Setup video writer if output path provided
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps,
                                          (frame_width, frame_height))
        
        # Reset analysis data
        self.frame_analysis = []
        emotion_counts = {emotion: 0 for emotion in EMOTION_LABELS}
        
        frame_num = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                display_frame = frame.copy()
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                frame_data = {
                    'frame_number': frame_num,
                    'timestamp': frame_num / fps,
                    'face_detected': landmarks is not None
                }
                
                if landmarks is not None:
                    # Calculate face bbox
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())
                    
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame_width, x_max + padding)
                    y_max = min(frame_height, y_max + padding)
                    
                    face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        face_tensor = self.preprocess_frame(face_img)
                        emotion, confidence, probs = self.detect_emotion(face_tensor, landmarks)
                        
                        emotion_counts[emotion] += 1
                        
                        # Store frame data
                        frame_data.update({
                            'emotion': emotion,
                            'confidence': float(confidence),
                            'probabilities': {EMOTION_LABELS[i]: float(probs[i])
                                            for i in range(len(EMOTION_LABELS))},
                            'face_bbox': face_bbox
                        })
                        
                        # Get attention weights
                        attention_map = self.get_attention_weights(face_tensor)
                        if attention_map is not None:
                            attention_scores = self.get_attention_focus_regions(landmarks, attention_map)
                            frame_data['attention_focus'] = attention_scores
                            
                            if self.show_attention:
                                self.overlay_attention_heatmap(display_frame, face_bbox, attention_map)
                        
                        # Draw landmarks
                        if self.show_landmarks:
                            highlight = []
                            if emotion in ['happy', 'sad', 'disgust']:
                                highlight.append('mouth_outer')
                            if emotion in ['surprise', 'fear', 'angry']:
                                highlight.extend(['left_eyebrow', 'right_eyebrow'])
                            if emotion in ['surprise', 'fear']:
                                highlight.extend(['left_eye', 'right_eye'])
                            
                            self.draw_landmarks(display_frame, landmarks, highlight)
                        
                        # Draw UI
                        self.draw_ui(display_frame, emotion, confidence, probs,
                                   frame_num, total_frames, face_bbox)
                    else:
                        self.draw_ui(display_frame, 'neutral', 0.0, np.zeros(7),
                                   frame_num, total_frames, None)
                else:
                    self.draw_ui(display_frame, 'neutral', 0.0, np.zeros(7),
                               frame_num, total_frames, None)
                
                self.frame_analysis.append(frame_data)
                
                # Write frame
                if video_writer:
                    video_writer.write(display_frame)
                
                # Display
                if display:
                    cv2.imshow('Processing Video', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_num % 30 == 0 or frame_num == total_frames:
                    elapsed = time.time() - start_time
                    fps_proc = frame_num / elapsed
                    eta = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
                    print(f"Progress: {frame_num}/{total_frames} ({frame_num/total_frames*100:.1f}%) "
                          f"| FPS: {fps_proc:.1f} | ETA: {eta:.1f}s", end='\r')
        
        finally:
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()
            self.face_mesh.close()
        
        # Compute summary
        total_time = time.time() - start_time
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        summary = {
            'input_file': input_path,
            'output_file': output_path,
            'processing_time': total_time,
            'total_frames': frame_num,
            'fps': fps,
            'frames_with_faces': sum(1 for f in self.frame_analysis if f['face_detected']),
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion
        }
        
        print(f"\n\n{'='*70}")
        print("Processing Complete!")
        print(f"{'='*70}")
        print(f"Frames processed: {frame_num}")
        print(f"Processing time: {total_time:.1f}s ({frame_num/total_time:.1f} FPS)")
        print(f"Frames with faces: {summary['frames_with_faces']}")
        print(f"Dominant emotion: {dominant_emotion}")
        print(f"\nEmotion Distribution:")
        for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / frame_num * 100 if frame_num > 0 else 0
            print(f"  {emotion:10s}: {count:4d} frames ({percentage:5.1f}%)")
        
        # Save analysis
        if self.save_analysis and output_path:
            analysis_path = output_path.rsplit('.', 1)[0] + '_analysis.json'
            with open(analysis_path, 'w') as f:
                json.dump({
                    'summary': summary,
                    'frame_data': self.frame_analysis
                }, f, indent=2)
            print(f"\n✓ Analysis saved to: {analysis_path}")
        
        if output_path:
            print(f"✓ Processed video saved to: {output_path}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description='Process Video with Facial Landmarks and Attention Analysis'
    )
    parser.add_argument('input', type=str, help='Input video file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video file path (default: input_processed.mp4)')
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--use-mediapipe', action='store_true',
                       help='Use MediaPipe-enhanced emotion detection model')
    parser.add_argument('--no-landmarks', action='store_true',
                       help='Disable landmark overlay')
    parser.add_argument('--no-attention', action='store_true',
                       help='Disable attention heatmap overlay')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display processing in real-time')
    parser.add_argument('--no-analysis', action='store_true',
                       help='Do not save frame analysis JSON')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    
    args = parser.parse_args()
    
    # Set default output path
    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_processed{ext}"
    
    # Create processor
    processor = VideoProcessor(
        model_path=args.model_path,
        use_mediapipe_model=args.use_mediapipe,
        device=args.device,
        show_landmarks=not args.no_landmarks,
        show_attention=not args.no_attention,
        save_analysis=not args.no_analysis
    )
    
    # Process video
    processor.process_video(
        input_path=args.input,
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()

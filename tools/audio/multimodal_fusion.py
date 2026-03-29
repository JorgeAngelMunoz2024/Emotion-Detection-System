"""
Multimodal Emotion Fusion: Audio + Visual
Combines audio sentiment analysis with visual emotion detection for comprehensive analysis.
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import torch
import numpy as np
import mediapipe as mp
from collections import deque
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
import os
import threading
import queue

try:
    import speech_recognition as sr
except ImportError:
    sr = None

from models.emotion_detector import HybridEmotionRecognizer, EMOTION_LABELS
from models.mediapipe_detector import MediaPipeEmotionDetector
from tools.audio.audio_sentiment_analyzer import AudioSentimentAnalyzer


class MultimodalEmotionFusion:
    """
    Fuses visual emotion detection with audio sentiment analysis.
    """
    def __init__(
        self,
        model_path: Optional[str] = None,
        use_mediapipe_model: bool = True,
        device: str = 'cpu',
        output_dir: str = 'multimodal_analysis',
        fusion_method: str = 'weighted'
    ):
        """
        Args:
            model_path: Path to visual emotion model
            use_mediapipe_model: Use MediaPipe-enhanced visual model
            device: 'cpu' or 'cuda'
            output_dir: Output directory for results
            fusion_method: 'weighted', 'average', 'max'
        """
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.fusion_method = fusion_method
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize visual emotion detector
        print("Loading visual emotion model...")
        if use_mediapipe_model:
            self.visual_model = MediaPipeEmotionDetector(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=True,
                fusion_method='concat'
            ).to(self.device)
        else:
            self.visual_model = HybridEmotionRecognizer(
                num_emotions=7,
                cnn_backbone='mobilenet_v2',
                pretrained=True,
                temporal_layers=2,
                temporal_heads=4,
                fusion_method='weighted'
            ).to(self.device)
        
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.visual_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded visual model from {model_path}")
        
        self.visual_model.eval()
        self.use_mediapipe_model = use_mediapipe_model
        
        # Initialize audio sentiment analyzer
        print("Loading audio sentiment analyzer...")
        self.audio_analyzer = AudioSentimentAnalyzer(
            use_transformer=True,
            output_dir=output_dir,
            chunk_duration=3
        )
        print("✓ Audio analyzer ready")
        
        # Initialize MediaPipe Face Mesh for visual
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Face detector backup
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Audio-visual mapping
        # Maps text sentiment to visual emotion compatibility
        self.sentiment_to_emotion_map = {
            'positive': {'happy': 0.8, 'surprise': 0.5, 'neutral': 0.3, 'sad': 0.1, 'angry': 0.1, 'fear': 0.1, 'disgust': 0.1},
            'negative': {'sad': 0.7, 'angry': 0.7, 'disgust': 0.6, 'fear': 0.5, 'neutral': 0.2, 'happy': 0.1, 'surprise': 0.2},
            'neutral': {'neutral': 0.8, 'happy': 0.3, 'sad': 0.3, 'surprise': 0.3, 'angry': 0.2, 'fear': 0.2, 'disgust': 0.2}
        }
        
        # Emotion colors (BGR)
        self.emotion_colors = {
            'angry': (0, 0, 255),
            'disgust': (0, 128, 128),
            'fear': (255, 0, 255),
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'surprise': (0, 255, 255),
            'neutral': (128, 128, 128)
        }
        
        # State
        self.audio_queue = queue.Queue()
        self.latest_audio_result = None
        self.audio_thread = None
        self.is_running = False
        self.frame_count = 0
        self.fusion_history = []
    
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
    
    def detect_visual_emotion(self, face_tensor: torch.Tensor,
                             landmarks: Optional[np.ndarray] = None) -> Tuple[str, float, np.ndarray]:
        """Detect emotion from visual input."""
        with torch.no_grad():
            if self.use_mediapipe_model and landmarks is not None:
                landmarks_tensor = torch.from_numpy(landmarks).unsqueeze(0).to(self.device)
                logits = self.visual_model(face_tensor, landmarks_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            else:
                logits = self.visual_model(face_tensor)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            emotion_idx = np.argmax(probs)
            confidence = probs[emotion_idx]
            emotion_label = EMOTION_LABELS[emotion_idx]
            
            return emotion_label, confidence, probs
    
    def audio_recording_worker(self):
        """Background thread for continuous audio recording."""
        if sr is None:
            print("Speech recognition not available")
            return
        
        microphone = sr.Microphone(sample_rate=16000)
        recognizer = sr.Recognizer()
        
        print("Audio recording started...")
        
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while self.is_running:
            try:
                audio_data = self.audio_analyzer.record_audio_chunk(microphone)
                if audio_data:
                    # Analyze in background
                    result = self.audio_analyzer.analyze_audio_chunk(audio_data, time.time())
                    self.latest_audio_result = result
                    self.audio_queue.put(result)
            except Exception as e:
                print(f"Audio recording error: {e}")
                time.sleep(0.5)
    
    def fuse_emotions(self, visual_probs: np.ndarray, audio_result: Optional[Dict]) -> Dict:
        """
        Fuse visual emotion probabilities with audio sentiment.
        
        Args:
            visual_probs: Visual emotion probabilities (7,)
            audio_result: Audio analysis result
            
        Returns:
            Fused emotion analysis
        """
        # If no audio, return visual only
        if audio_result is None or audio_result['text'] is None:
            emotion_idx = np.argmax(visual_probs)
            return {
                'emotion': EMOTION_LABELS[emotion_idx],
                'confidence': float(visual_probs[emotion_idx]),
                'probabilities': {EMOTION_LABELS[i]: float(visual_probs[i]) for i in range(7)},
                'visual_emotion': EMOTION_LABELS[emotion_idx],
                'audio_sentiment': None,
                'fusion_method': 'visual_only'
            }
        
        # Get audio sentiment
        audio_sentiment = audio_result['sentiment']['sentiment']
        audio_confidence = audio_result['sentiment'].get('confidence', 0.5)
        
        # Get sentiment-emotion compatibility weights
        compatibility = self.sentiment_to_emotion_map[audio_sentiment]
        
        # Fuse probabilities
        if self.fusion_method == 'weighted':
            # Weighted average with audio boosting compatible emotions
            visual_weight = 0.6
            audio_weight = 0.4
            
            fused_probs = np.zeros(7)
            for i, emotion in enumerate(EMOTION_LABELS):
                visual_score = visual_probs[i]
                audio_boost = compatibility.get(emotion, 0.2) * audio_confidence
                fused_probs[i] = visual_weight * visual_score + audio_weight * audio_boost
            
            # Normalize
            fused_probs = fused_probs / fused_probs.sum()
            
        elif self.fusion_method == 'average':
            # Simple average
            audio_probs = np.array([compatibility.get(e, 0.2) for e in EMOTION_LABELS])
            audio_probs = audio_probs / audio_probs.sum()
            fused_probs = (visual_probs + audio_probs) / 2
            
        else:  # max
            # Take maximum confidence
            audio_probs = np.array([compatibility.get(e, 0.2) for e in EMOTION_LABELS])
            fused_probs = np.maximum(visual_probs, audio_probs)
            fused_probs = fused_probs / fused_probs.sum()
        
        emotion_idx = np.argmax(fused_probs)
        
        return {
            'emotion': EMOTION_LABELS[emotion_idx],
            'confidence': float(fused_probs[emotion_idx]),
            'probabilities': {EMOTION_LABELS[i]: float(fused_probs[i]) for i in range(7)},
            'visual_emotion': EMOTION_LABELS[np.argmax(visual_probs)],
            'audio_sentiment': audio_sentiment,
            'audio_text': audio_result['text'],
            'audio_confidence': audio_confidence,
            'fusion_method': self.fusion_method
        }
    
    def draw_multimodal_ui(self, frame: np.ndarray, fusion_result: Dict, 
                          face_bbox: Optional[Tuple], fps: float):
        """Draw UI with both visual and audio information."""
        h, w = frame.shape[:2]
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Face and emotion
        if face_bbox:
            x, y, w_box, h_box = face_bbox
            color = self.emotion_colors[fusion_result['emotion']]
            cv2.rectangle(frame, (x, y), (x + w_box, y + h_box), color, 2)
            
            label = f"{fusion_result['emotion']}: {fusion_result['confidence']:.2f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Multimodal panel
        panel_y = 60
        
        # Visual emotion
        visual_text = f"Visual: {fusion_result['visual_emotion']}"
        cv2.putText(frame, visual_text, (10, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Audio sentiment
        if fusion_result['audio_sentiment']:
            panel_y += 25
            sentiment = fusion_result['audio_sentiment']
            sentiment_color = (0, 255, 0) if sentiment == 'positive' else ((0, 0, 255) if sentiment == 'negative' else (128, 128, 128))
            audio_text = f"Audio: {sentiment} ({fusion_result['audio_confidence']:.2f})"
            cv2.putText(frame, audio_text, (10, panel_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, sentiment_color, 1)
            
            # Latest transcribed text
            if fusion_result.get('audio_text'):
                panel_y += 25
                text_preview = fusion_result['audio_text'][:50] + "..." if len(fusion_result['audio_text']) > 50 else fusion_result['audio_text']
                cv2.putText(frame, f'"{text_preview}"', (10, panel_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Emotion bars (fused probabilities)
        bar_height = 18
        bar_width = 150
        start_x = w - bar_width - 20
        start_y = 60
        
        cv2.putText(frame, "Fused Emotions", (start_x, start_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        for i, (emotion_label, prob) in enumerate(fusion_result['probabilities'].items()):
            y_pos = start_y + i * (bar_height + 3)
            
            # Background
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + bar_width, y_pos + bar_height),
                         (50, 50, 50), -1)
            
            # Bar
            prob_width = int(bar_width * prob)
            color = self.emotion_colors[emotion_label]
            cv2.rectangle(frame, (start_x, y_pos),
                         (start_x + prob_width, y_pos + bar_height),
                         color, -1)
            
            # Label
            text = f"{emotion_label}: {prob:.2f}"
            cv2.putText(frame, text, (start_x + 5, y_pos + 13),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    
    def run(self):
        """Run real-time multimodal emotion analysis."""
        print("=" * 70)
        print("Multimodal Emotion Analysis (Visual + Audio)")
        print("=" * 70)
        print("Starting webcam and audio recording...")
        print("Press 'Q' to quit, 'S' to save analysis\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Start audio recording thread
        self.is_running = True
        self.audio_thread = threading.Thread(target=self.audio_recording_worker, daemon=True)
        self.audio_thread.start()
        
        fps_counter = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                display_frame = frame.copy()
                self.frame_count += 1
                
                # Extract landmarks
                landmarks = self.extract_landmarks(frame)
                
                fusion_result = None
                face_bbox = None
                
                if landmarks is not None:
                    # Get face bbox
                    x_coords = landmarks[:, 0]
                    y_coords = landmarks[:, 1]
                    x_min, x_max = int(x_coords.min()), int(x_coords.max())
                    y_min, y_max = int(y_coords.min()), int(y_coords.max())
                    
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame.shape[1], x_max + padding)
                    y_max = min(frame.shape[0], y_max + padding)
                    
                    face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        face_tensor = self.preprocess_frame(face_img)
                        _, _, visual_probs = self.detect_visual_emotion(face_tensor, landmarks)
                        
                        # Fuse with latest audio
                        fusion_result = self.fuse_emotions(visual_probs, self.latest_audio_result)
                        self.fusion_history.append({
                            'frame': self.frame_count,
                            'timestamp': time.time(),
                            **fusion_result
                        })
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_counter.append(fps)
                avg_fps = np.mean(fps_counter)
                
                # Draw UI
                if fusion_result:
                    self.draw_multimodal_ui(display_frame, fusion_result, face_bbox, avg_fps)
                
                cv2.imshow('Multimodal Emotion Analysis', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_results()
        
        finally:
            self.is_running = False
            if self.audio_thread:
                self.audio_thread.join(timeout=2)
            
            cap.release()
            cv2.destroyAllWindows()
            self.face_mesh.close()
            
            print("\n✓ Multimodal analysis stopped")
            self.save_results()
    
    def save_results(self):
        """Save analysis results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"multimodal_{timestamp}.json")
        
        data = {
            'fusion_history': self.fusion_history,
            'audio_history': list(self.audio_analyzer.sentiment_history),
            'summary': self.generate_summary()
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, indent=2, fp=f)
        
        print(f"✓ Results saved to: {output_file}")
    
    def generate_summary(self) -> Dict:
        """Generate analysis summary."""
        if not self.fusion_history:
            return {}
        
        emotion_counts = {e: 0 for e in EMOTION_LABELS}
        for item in self.fusion_history:
            emotion_counts[item['emotion']] += 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        return {
            'total_frames': len(self.fusion_history),
            'emotion_distribution': emotion_counts,
            'dominant_emotion': dominant_emotion,
            'fusion_method': self.fusion_method
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multimodal Emotion Analysis (Visual + Audio)'
    )
    parser.add_argument('--model-path', type=str, default=None,
                       help='Path to visual emotion model')
    parser.add_argument('--use-mediapipe', action='store_true',
                       help='Use MediaPipe-enhanced visual model')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'], help='Device for inference')
    parser.add_argument('--output-dir', type=str, default='multimodal_analysis',
                       help='Output directory')
    parser.add_argument('--fusion', type=str, default='weighted',
                       choices=['weighted', 'average', 'max'],
                       help='Fusion method')
    
    args = parser.parse_args()
    
    fusion = MultimodalEmotionFusion(
        model_path=args.model_path,
        use_mediapipe_model=args.use_mediapipe,
        device=args.device,
        output_dir=args.output_dir,
        fusion_method=args.fusion
    )
    
    fusion.run()


if __name__ == "__main__":
    main()

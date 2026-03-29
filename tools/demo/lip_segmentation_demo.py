"""
Lip Segmentation-based Emotion Detection Demo
==============================================

Real-time emotion detection using lip segmentation instead of MediaPipe landmarks.
This provides an alternative approach to facial emotion recognition.

Usage:
    python lip_segmentation_demo.py [--model-path PATH] [--camera 0]
"""

import cv2
import numpy as np
import argparse
import sys
import os
from pathlib import Path
import time
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.lip_segmentation_detector import LipSegmentationDetector


class LipSegmentationDemo:
    """
    Interactive demo for lip segmentation-based emotion detection
    """
    
    def __init__(self, model_path=None, camera_id=0, show_segmentation=True):
        """
        Initialize demo
        
        Args:
            model_path: Path to pre-trained BiSeNet weights
            camera_id: Camera device ID
            show_segmentation: Whether to show segmentation overlay
        """
        self.camera_id = camera_id
        self.show_segmentation = show_segmentation
        
        # Initialize detector
        print("Initializing Lip Segmentation Detector...")
        self.detector = LipSegmentationDetector(
            model_path=model_path,
            device='cpu'  # Change to 'cuda' if GPU available
        )
        print("✓ Detector initialized")
        
        # Video capture
        self.cap = None
        
        # FPS tracking
        self.fps_history = deque(maxlen=30)
        self.last_time = time.time()
        
        # UI settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.emotion_colors = {
            'happy': (0, 255, 0),
            'sad': (255, 0, 0),
            'angry': (0, 0, 255),
            'surprise': (255, 255, 0),
            'neutral': (200, 200, 200),
            'disgust': (128, 0, 128),
            'fear': (255, 165, 0)
        }
    
    def initialize_camera(self):
        """Initialize camera capture"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.camera_id}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"✓ Camera {self.camera_id} initialized")
    
    def calculate_fps(self):
        """Calculate and return current FPS"""
        current_time = time.time()
        fps = 1.0 / (current_time - self.last_time + 1e-6)
        self.last_time = current_time
        self.fps_history.append(fps)
        
        return np.mean(self.fps_history) if self.fps_history else 0
    
    def draw_ui(self, frame, results):
        """
        Draw UI elements on frame
        
        Args:
            frame: Input frame
            results: Detection results from detector
            
        Returns:
            Frame with UI overlay
        """
        display_frame = frame.copy()
        h, w = frame.shape[:2]
        
        # Draw segmentation overlay if enabled
        if self.show_segmentation and 'segmentation' in results:
            display_frame = self.detector.visualize_segmentation(
                display_frame, 
                results['segmentation'],
                alpha=0.3
            )
        
        # Draw lip bounding box
        lip_features = results.get('lip_features', {})
        if 'lip_bbox' in lip_features:
            x, y, bw, bh = lip_features['lip_bbox']
            cv2.rectangle(display_frame, (x, y), (x + bw, y + bh), (0, 255, 255), 2)
            
            # Draw lip center
            if 'lip_center' in lip_features:
                cx, cy = lip_features['lip_center']
                cv2.circle(display_frame, (cx, cy), 3, (0, 255, 255), -1)
        
        # Draw emotion label
        emotion = results.get('emotion', 'unknown')
        confidence = results.get('confidence', 0.0)
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Emotion text with background
        text = f"{emotion.upper()}"
        text_size = cv2.getTextSize(text, self.font, 1.5, 3)[0]
        
        # Draw background rectangle
        cv2.rectangle(display_frame, 
                     (10, 10), 
                     (text_size[0] + 20, text_size[1] + 30),
                     color, 
                     -1)
        
        # Draw text
        cv2.putText(display_frame, text, (15, text_size[1] + 20), 
                   self.font, 1.5, (0, 0, 0), 3)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = text_size[1] + 50
        
        cv2.rectangle(display_frame, 
                     (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), 
                     -1)
        
        filled_width = int(bar_width * confidence)
        cv2.rectangle(display_frame,
                     (bar_x, bar_y),
                     (bar_x + filled_width, bar_y + bar_height),
                     color,
                     -1)
        
        cv2.putText(display_frame, f"Confidence: {confidence:.2f}",
                   (bar_x, bar_y - 5), self.font, 0.5, (255, 255, 255), 1)
        
        # Draw lip features
        y_offset = bar_y + bar_height + 30
        
        if lip_features:
            feature_texts = [
                f"Lip Area: {lip_features.get('lip_area', 0):.0f}",
                f"Aspect Ratio: {lip_features.get('lip_aspect_ratio', 0):.2f}",
                f"Mouth Openness: {lip_features.get('mouth_openness', 0):.2f}",
                f"Upper/Lower: {lip_features.get('upper_lower_ratio', 0):.2f}"
            ]
            
            for i, text in enumerate(feature_texts):
                cv2.putText(display_frame, text,
                           (10, y_offset + i * 25),
                           self.font, 0.5, (255, 255, 255), 1)
        
        # Draw FPS
        fps = self.calculate_fps()
        cv2.putText(display_frame, f"FPS: {fps:.1f}",
                   (w - 120, 30), self.font, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "ESC: Exit",
            "S: Toggle Segmentation",
            "SPACE: Save Frame"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(display_frame, instruction,
                       (w - 220, h - 60 + i * 25),
                       self.font, 0.4, (255, 255, 255), 1)
        
        return display_frame
    
    def run(self):
        """Run the demo"""
        try:
            self.initialize_camera()
            
            print("\n" + "="*60)
            print("Lip Segmentation Emotion Detection Demo")
            print("="*60)
            print("\nControls:")
            print("  ESC      - Exit demo")
            print("  S        - Toggle segmentation overlay")
            print("  SPACE    - Save current frame")
            print("\nStarting detection...")
            print("="*60 + "\n")
            
            frame_count = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Flip frame horizontally (mirror view)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                results = self.detector.process_frame(frame)
                
                # Draw UI
                display_frame = self.draw_ui(frame, results)
                
                # Show frame
                cv2.imshow('Lip Segmentation Emotion Detection', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27:  # ESC
                    print("\nExiting...")
                    break
                elif key == ord('s') or key == ord('S'):
                    self.show_segmentation = not self.show_segmentation
                    print(f"Segmentation overlay: {'ON' if self.show_segmentation else 'OFF'}")
                elif key == ord(' '):  # SPACE
                    filename = f"frame_{frame_count:04d}.png"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved frame: {filename}")
                    frame_count += 1
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            # Cleanup
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo ended.")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Lip Segmentation-based Emotion Detection Demo"
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to pre-trained BiSeNet weights (optional)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--no-segmentation',
        action='store_true',
        help='Disable segmentation overlay'
    )
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = LipSegmentationDemo(
        model_path=args.model_path,
        camera_id=args.camera,
        show_segmentation=not args.no_segmentation
    )
    
    demo.run()


if __name__ == '__main__':
    main()

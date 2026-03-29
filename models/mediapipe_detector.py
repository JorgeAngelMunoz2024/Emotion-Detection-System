"""
Enhanced Emotion Detection with MediaPipe Face Mesh
Combines CNN spatial features with facial landmark information for improved accuracy.

MediaPipe provides 468 facial landmarks focusing on:
- Eyes (left/right: 33 landmarks each)
- Eyebrows (20 landmarks each)
- Nose (21 landmarks)
- Mouth (40 landmarks)
- Face oval (36 landmarks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import cv2

from .emotion_detector import SpatialAttentionCNN, EMOTION_LABELS


class FaceMeshFeatureExtractor(nn.Module):
    """
    Extracts and processes facial landmark features from MediaPipe Face Mesh.
    Focuses on emotion-relevant regions: eyes, eyebrows, mouth, nose.
    """
    def __init__(self, landmark_dim: int = 468 * 3, hidden_dim: int = 256):
        """
        Args:
            landmark_dim: Dimension of flattened landmarks (468 points * 3 coords = 1404)
            hidden_dim: Hidden dimension for feature processing
        """
        super(FaceMeshFeatureExtractor, self).__init__()
        
        # Process raw landmark coordinates
        self.landmark_encoder = nn.Sequential(
            nn.Linear(landmark_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Specific region encoders
        # Eyes (left: 33-133, right: 362-263)
        self.eye_encoder = nn.Sequential(
            nn.Linear(66 * 3, 128),  # 66 points * 3 coords
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Eyebrows (left: 55-65, right: 285-295)
        self.eyebrow_encoder = nn.Sequential(
            nn.Linear(40 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Mouth (lips: 61, 146, 91, 181, etc.)
        self.mouth_encoder = nn.Sequential(
            nn.Linear(40 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Combine all region features
        self.region_fusion = nn.Sequential(
            nn.Linear(64 + 32 + 64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
    def extract_region_landmarks(self, landmarks: np.ndarray) -> dict:
        """
        Extract specific facial regions from 468 landmarks.
        
        Args:
            landmarks: (468, 3) array of facial landmarks
            
        Returns:
            Dictionary with region-specific landmarks
        """
        # Key landmark indices for emotion-relevant regions
        left_eye_indices = list(range(33, 133))  # Simplified
        right_eye_indices = list(range(362, 398))
        left_eyebrow_indices = list(range(55, 65))
        right_eyebrow_indices = list(range(285, 295))
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
                        291, 308, 324, 318, 402, 317, 14, 87, 178, 88,
                        95, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                        415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
        
        return {
            'left_eye': landmarks[left_eye_indices[:33]].flatten(),
            'right_eye': landmarks[right_eye_indices[:33]].flatten(),
            'left_eyebrow': landmarks[left_eyebrow_indices[:20]].flatten(),
            'right_eyebrow': landmarks[right_eyebrow_indices[:20]].flatten(),
            'mouth': landmarks[mouth_indices[:40]].flatten()
        }
    
    def forward(self, landmarks: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Process facial landmarks into feature representations.
        
        Args:
            landmarks: (B, 468, 3) facial landmarks
            
        Returns:
            (global_features, region_features)
        """
        B = landmarks.shape[0]
        
        # Flatten landmarks for global processing
        landmarks_flat = landmarks.view(B, -1)
        global_features = self.landmark_encoder(landmarks_flat)
        
        # Extract and process region-specific features
        # Note: This is simplified - in practice, use proper indexing
        region_features = {}
        
        # For now, use placeholder slicing (update with proper indices)
        eye_features = self.eye_encoder(landmarks_flat[:, :66*3])
        eyebrow_features = self.eyebrow_encoder(landmarks_flat[:, 66*3:106*3])
        mouth_features = self.mouth_encoder(landmarks_flat[:, 106*3:146*3])
        
        # Combine region features
        combined_regions = torch.cat([eye_features, eyebrow_features, mouth_features], dim=1)
        region_features = self.region_fusion(combined_regions)
        
        return global_features, region_features


class MediaPipeEmotionDetector(nn.Module):
    """
    Enhanced emotion detector combining CNN image features with MediaPipe facial landmarks.
    
    Architecture:
    1. CNN: Extract spatial features from face image
    2. MediaPipe: Extract 468 facial landmarks
    3. Fusion: Combine CNN + landmark features
    4. Classification: Predict emotion
    """
    def __init__(
        self,
        num_emotions: int = 7,
        cnn_backbone: str = 'mobilenet_v2',
        pretrained: bool = True,
        fusion_method: str = 'concat',  # 'concat', 'add', 'attention'
        dropout: float = 0.4
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            cnn_backbone: CNN backbone for image features
            pretrained: Use pretrained CNN weights
            fusion_method: How to fuse CNN and landmark features
            dropout: Dropout rate
        """
        super(MediaPipeEmotionDetector, self).__init__()
        
        self.num_emotions = num_emotions
        self.fusion_method = fusion_method
        
        # CNN for image features
        self.cnn = SpatialAttentionCNN(
            num_emotions=num_emotions,
            backbone=cnn_backbone,
            pretrained=pretrained,
            use_attention=True,
            dropout=dropout
        )
        
        # Get CNN feature dimension
        cnn_feature_dim = self.cnn.feature_dim
        
        # MediaPipe landmark feature extractor
        self.landmark_extractor = FaceMeshFeatureExtractor(
            landmark_dim=468 * 3,
            hidden_dim=256
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            fusion_dim = cnn_feature_dim + 256 + 256  # CNN + global landmarks + region landmarks
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, num_emotions)
            )
        elif fusion_method == 'attention':
            # Cross-attention between CNN and landmark features
            self.cnn_proj = nn.Linear(cnn_feature_dim, 256)
            self.landmark_proj = nn.Linear(512, 256)  # global + region
            self.cross_attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
            self.fusion = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_emotions)
            )
        else:  # 'add'
            self.cnn_proj = nn.Linear(cnn_feature_dim, 256)
            self.landmark_proj = nn.Linear(512, 256)
            self.fusion = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_emotions)
            )
    
    def forward(
        self, 
        image: torch.Tensor, 
        landmarks: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass combining image and landmark features.
        
        Args:
            image: Face images (B, 3, H, W)
            landmarks: Facial landmarks (B, 468, 3)
            return_features: If True, return intermediate features
            
        Returns:
            Emotion logits (B, num_emotions) or features if return_features=True
        """
        # Extract CNN features
        cnn_features = self.cnn.extract_features(image)
        
        # Extract landmark features
        landmark_global, landmark_region = self.landmark_extractor(landmarks)
        landmark_features = torch.cat([landmark_global, landmark_region], dim=1)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([cnn_features, landmark_features], dim=1)
            logits = self.fusion(fused)
        elif self.fusion_method == 'attention':
            cnn_proj = self.cnn_proj(cnn_features).unsqueeze(1)
            landmark_proj = self.landmark_proj(landmark_features).unsqueeze(1)
            attended, _ = self.cross_attention(cnn_proj, landmark_proj, landmark_proj)
            logits = self.fusion(attended.squeeze(1))
        else:  # 'add'
            cnn_proj = self.cnn_proj(cnn_features)
            landmark_proj = self.landmark_proj(landmark_features)
            fused = cnn_proj + landmark_proj
            logits = self.fusion(fused)
        
        if return_features:
            return logits, cnn_features, landmark_features
        
        return logits
    
    def predict_emotion(
        self, 
        image: torch.Tensor, 
        landmarks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion with probabilities.
        
        Returns:
            (predicted_class, probabilities)
        """
        logits = self.forward(image, landmarks)
        probs = F.softmax(logits, dim=1)
        predicted = torch.argmax(probs, dim=1)
        return predicted, probs


class MediaPipeFaceDetector:
    """
    Wrapper for MediaPipe Face Mesh detection.
    Extracts 468 facial landmarks for emotion analysis.
    """
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect facial landmarks from image.
        
        Args:
            image: BGR image from OpenCV
            
        Returns:
            (468, 3) array of landmarks or None if no face detected
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array (468, 3)
        h, w = image.shape[:2]
        landmarks = np.array([
            [lm.x * w, lm.y * h, lm.z * w] 
            for lm in face_landmarks.landmark
        ])
        
        return landmarks
    
    def draw_landmarks(self, image: np.ndarray, landmarks: np.ndarray):
        """
        Draw facial landmarks on image.
        
        Args:
            image: BGR image
            landmarks: (468, 3) landmarks array
        """
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)
    
    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test MediaPipe + CNN model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    print("=" * 60)
    print("Testing MediaPipe Emotion Detector")
    print("=" * 60)
    
    # Create model
    model = MediaPipeEmotionDetector(
        num_emotions=7,
        cnn_backbone='mobilenet_v2',
        pretrained=True,
        fusion_method='concat',
        dropout=0.4
    ).to(device)
    
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test with dummy data
    test_image = torch.randn(2, 3, 224, 224).to(device)
    test_landmarks = torch.randn(2, 468, 3).to(device)
    
    output = model(test_image, test_landmarks)
    print(f"Input image: {test_image.shape}")
    print(f"Input landmarks: {test_landmarks.shape}")
    print(f"Output: {output.shape}")
    
    predicted, probs = model.predict_emotion(test_image, test_landmarks)
    print(f"Predicted emotions: {predicted}")
    print(f"Probabilities shape: {probs.shape}")
    
    print("\n" + "=" * 60)
    print("✓ MediaPipe model test passed!")
    print("=" * 60)
    
    # Test MediaPipe detector
    print("\nTesting MediaPipe Face Detector...")
    detector = MediaPipeFaceDetector()
    
    # Create dummy image
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    landmarks = detector.detect_landmarks(dummy_img)
    
    if landmarks is not None:
        print(f"Detected landmarks shape: {landmarks.shape}")
        print("✓ MediaPipe detector test passed!")
    else:
        print("No face detected (expected with random image)")
    
    detector.close()
    
    print("\nRecommendation: MediaPipe provides precise facial landmarks")
    print("for emotion-relevant regions (eyes, eyebrows, mouth)!")

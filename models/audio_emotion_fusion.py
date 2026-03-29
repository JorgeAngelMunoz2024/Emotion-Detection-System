"""
Audio-Enhanced Emotion Recognition
Integrates audio sentiment features into the transformer architecture for multimodal fusion.
Supports: Visual (CNN) + Temporal (Transformer) + Landmarks (MediaPipe) + Audio (NLP)
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
import numpy as np

from models.emotion_detector import SpatialAttentionCNN, TemporalTransformer, EMOTION_LABELS
from models.audio_acoustic_encoder import AcousticEmotionEncoder
from models.personalized_acoustic_profiling import (
    PersonalizedAcousticEncoder,
    PersonalizedAcousticProfileManager
)

try:
    from models.mediapipe_detector import FaceMeshFeatureExtractor
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


class AudioFeatureEncoder(nn.Module):
    """
    Encodes audio sentiment features for integration with visual features.
    Takes sentiment scores, emotion probabilities, and text embeddings.
    """
    def __init__(
        self,
        num_emotions: int = 7,
        sentiment_dim: int = 3,  # positive, negative, neutral
        hidden_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Args:
            num_emotions: Number of emotion classes
            sentiment_dim: Dimension of sentiment features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
        """
        super(AudioFeatureEncoder, self).__init__()
        
        self.num_emotions = num_emotions
        self.sentiment_dim = sentiment_dim
        
        # Encode sentiment scores
        self.sentiment_encoder = nn.Sequential(
            nn.Linear(sentiment_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Encode emotion keyword features
        self.emotion_encoder = nn.Sequential(
            nn.Linear(num_emotions, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128)
        )
        
        # Combine sentiment + emotion features
        self.fusion_encoder = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Audio-to-emotion classifier
        self.audio_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, sentiment_scores: torch.Tensor, 
                emotion_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode audio features.
        
        Args:
            sentiment_scores: Sentiment probabilities (B, 3) [positive, negative, neutral]
            emotion_scores: Emotion keyword scores (B, 7)
            
        Returns:
            (audio_features, audio_logits)
            - audio_features: Encoded audio features (B, hidden_dim)
            - audio_logits: Direct audio emotion prediction (B, num_emotions)
        """
        # Encode components
        sentiment_feat = self.sentiment_encoder(sentiment_scores)
        emotion_feat = self.emotion_encoder(emotion_scores)
        
        # Combine
        combined = torch.cat([sentiment_feat, emotion_feat], dim=1)
        audio_features = self.fusion_encoder(combined)
        
        # Predict emotion from audio
        audio_logits = self.audio_classifier(audio_features)
        
        return audio_features, audio_logits


class AudioVisualTransformer(nn.Module):
    """
    Transformer that processes both visual and audio features jointly.
    Enables cross-modal attention between video frames and audio segments.
    """
    def __init__(
        self,
        visual_dim: int,
        audio_dim: int,
        num_emotions: int = 7,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Args:
            visual_dim: Dimension of visual features
            audio_dim: Dimension of audio features
            num_emotions: Number of emotion classes
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(AudioVisualTransformer, self).__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.num_emotions = num_emotions
        
        # Project features to common dimension
        embed_dim = max(visual_dim, audio_dim)
        self.visual_projection = nn.Linear(visual_dim, embed_dim)
        self.audio_projection = nn.Linear(audio_dim, embed_dim)
        
        # Modal type embeddings (to distinguish visual from audio)
        self.modal_embedding = nn.Embedding(2, embed_dim)  # 0=visual, 1=audio
        
        # Positional encoding for temporal sequence
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 100, embed_dim)  # Max 100 frames/segments
        )
        
        # Transformer encoder for joint processing
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Separate attention for visual and audio
        self.visual_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-modal attention (visual attends to audio)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),  # *2 for visual + audio pooled features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, visual_features: torch.Tensor, 
                audio_features: torch.Tensor) -> torch.Tensor:
        """
        Process visual and audio features jointly.
        
        Args:
            visual_features: Visual feature sequence (B, T_v, visual_dim)
            audio_features: Audio feature sequence (B, T_a, audio_dim)
            
        Returns:
            Emotion logits (B, num_emotions)
        """
        B = visual_features.shape[0]
        T_v = visual_features.shape[1]
        T_a = audio_features.shape[1]
        
        # Project to common dimension
        visual_proj = self.visual_projection(visual_features)  # (B, T_v, embed_dim)
        audio_proj = self.audio_projection(audio_features)      # (B, T_a, embed_dim)
        
        # Add modal type embeddings
        visual_modal = self.modal_embedding(torch.zeros(B, T_v, dtype=torch.long, device=visual_features.device))
        audio_modal = self.modal_embedding(torch.ones(B, T_a, dtype=torch.long, device=audio_features.device))
        
        visual_proj = visual_proj + visual_modal
        audio_proj = audio_proj + audio_modal
        
        # Add positional encoding
        visual_proj = visual_proj + self.positional_encoding[:, :T_v, :]
        audio_proj = audio_proj + self.positional_encoding[:, :T_a, :]
        
        # Self-attention within each modality
        visual_attended, _ = self.visual_attention(visual_proj, visual_proj, visual_proj)
        audio_attended, _ = self.audio_attention(audio_proj, audio_proj, audio_proj)
        
        # Cross-modal attention (visual queries, audio keys/values)
        cross_attended, _ = self.cross_attention(visual_attended, audio_attended, audio_attended)
        
        # Concatenate all features
        combined = torch.cat([visual_proj, audio_proj], dim=1)  # (B, T_v+T_a, embed_dim)
        
        # Joint transformer processing
        joint_features = self.transformer(combined)  # (B, T_v+T_a, embed_dim)
        
        # Pool visual and audio separately
        visual_pooled = joint_features[:, :T_v, :].mean(dim=1)  # (B, embed_dim)
        audio_pooled = joint_features[:, T_v:, :].mean(dim=1)   # (B, embed_dim)
        
        # Combine and classify
        final_features = torch.cat([visual_pooled, audio_pooled], dim=1)  # (B, embed_dim*2)
        logits = self.classifier(final_features)
        
        return logits


class MultimodalEmotionRecognizer(nn.Module):
    """
    Complete multimodal emotion recognition system.
    Integrates: Visual (CNN) + Temporal (Transformer) + Landmarks + Audio (NLP)
    
    Architecture:
    1. Visual CNN: Extract spatial features from frames
    2. Temporal Transformer: Model dynamics over time
    3. MediaPipe (optional): Facial landmark geometry
    4. Audio Encoder: Sentiment and emotion from speech
    5. Audio-Visual Transformer: Joint cross-modal processing
    6. Multimodal Fusion: Combine all modalities
    """
    def __init__(
        self,
        num_emotions: int = 7,
        cnn_backbone: str = 'mobilenet_v2',
        pretrained: bool = True,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        use_mediapipe: bool = False,
        use_audio: bool = True,
        use_acoustic: bool = True,
        use_personalized_acoustic: bool = True,
        profile_manager: Optional[PersonalizedAcousticProfileManager] = None,
        fusion_method: str = 'weighted',  # 'weighted', 'concat', 'attention'
        dropout: float = 0.3
    ):
        """
        Args:
            num_emotions: Number of emotion classes (default: 7)
            cnn_backbone: CNN backbone ('mobilenet_v2', 'resnet18', etc.)
            pretrained: Use pretrained CNN weights
            temporal_layers: Number of transformer layers
            temporal_heads: Number of attention heads
            use_mediapipe: Include facial landmark features
            use_audio: Include audio sentiment features (NLP)
            use_acoustic: Include acoustic features (spectrogram/prosody)
            use_personalized_acoustic: Use personalized acoustic profiling (person-specific)
            profile_manager: PersonalizedAcousticProfileManager for managing user profiles
            fusion_method: Multimodal fusion strategy
            dropout: Dropout rate
        """
        super(MultimodalEmotionRecognizer, self).__init__()
        
        self.num_emotions = num_emotions
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        self.use_audio = use_audio
        self.use_acoustic = use_acoustic
        self.use_personalized_acoustic = use_personalized_acoustic
        self.fusion_method = fusion_method
        
        # Profile manager for personalized acoustics
        if profile_manager is None and use_personalized_acoustic:
            self.profile_manager = PersonalizedAcousticProfileManager(
                profiles_dir="./acoustic_profiles"
            )
        else:
            self.profile_manager = profile_manager
        
        # Visual CNN (spatial features)
        self.spatial_cnn = SpatialAttentionCNN(
            num_emotions=num_emotions,
            backbone=cnn_backbone,
            pretrained=pretrained,
            use_attention=True,
            dropout=dropout
        )
        
        visual_dim = self.spatial_cnn.feature_dim
        
        # Temporal Transformer (video-only)
        self.temporal_transformer = TemporalTransformer(
            feature_dim=visual_dim,
            num_emotions=num_emotions,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            dropout=dropout
        )
        
        # MediaPipe landmarks (optional)
        if self.use_mediapipe:
            self.landmark_extractor = FaceMeshFeatureExtractor(
                landmark_dim=468 * 3,
                hidden_dim=256
            )
            self.landmark_classifier = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_emotions)
            )
        
        # Audio encoder (NLP sentiment)
        if self.use_audio:
            self.audio_encoder = AudioFeatureEncoder(
                num_emotions=num_emotions,
                sentiment_dim=3,
                hidden_dim=256,
                dropout=dropout
            )
            
            # Audio-Visual joint transformer
            self.audiovisual_transformer = AudioVisualTransformer(
                visual_dim=visual_dim,
                audio_dim=256,
                num_emotions=num_emotions,
                num_layers=temporal_layers,
                num_heads=temporal_heads,
                dropout=dropout
            )
        
        # Acoustic encoder (spectrogram/prosody for tone)
        if self.use_acoustic:
            if self.use_personalized_acoustic:
                # Personalized acoustic encoder with person-specific adaptation
                self.acoustic_encoder = PersonalizedAcousticEncoder(
                    n_mels=128,
                    sample_rate=16000,
                    hidden_dim=256,
                    num_emotions=num_emotions,
                    dropout=dropout
                )
            else:
                # Base acoustic encoder (no personalization)
                self.acoustic_encoder = AcousticEmotionEncoder(
                    n_mels=128,
                    sample_rate=16000,
                    hidden_dim=256,
                    num_emotions=num_emotions,
                    dropout=dropout
                )
        
        # Count modalities
        num_modalities = 2  # CNN + Transformer (base)
        if self.use_mediapipe:
            num_modalities += 1
        if self.use_audio:
            num_modalities += 2  # Audio + AudioVisual joint
        if self.use_acoustic:
            num_modalities += 1  # Acoustic
        
        # Fusion
        if fusion_method == 'weighted':
            self.fusion_weights = nn.Parameter(torch.ones(num_modalities) / num_modalities)
        elif fusion_method == 'concat':
            self.fusion_fc = nn.Sequential(
                nn.Linear(num_emotions * num_modalities, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_emotions)
            )
        elif fusion_method == 'attention':
            # Learn attention weights for each modality
            self.modality_attention = nn.Sequential(
                nn.Linear(num_emotions, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
    
    def forward_visual_only(self, frame_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Visual-only forward pass (no audio, no landmarks)."""
        B, T, C, H, W = frame_sequence.shape
        
        # Extract features from each frame
        features_list = []
        spatial_logits_list = []
        
        for t in range(T):
            frame = frame_sequence[:, t, :, :, :]
            features = self.spatial_cnn.extract_features(frame)
            features_list.append(features)
            spatial_logits = self.spatial_cnn.classifier(features)
            spatial_logits_list.append(spatial_logits)
        
        features_sequence = torch.stack(features_list, dim=1)
        spatial_logits = torch.stack(spatial_logits_list, dim=1).mean(dim=1)
        
        # Temporal
        temporal_logits = self.temporal_transformer(features_sequence)
        
        # Fusion
        if self.fusion_method == 'weighted':
            w = F.softmax(self.fusion_weights[:2], dim=0)
            combined_logits = w[0] * spatial_logits + w[1] * temporal_logits
        elif self.fusion_method == 'concat':
            concat_logits = torch.cat([spatial_logits, temporal_logits], dim=1)
            combined_logits = self.fusion_fc(concat_logits)
        else:
            combined_logits = (spatial_logits + temporal_logits) / 2
        
        return {
            'logits': combined_logits,
            'spatial_logits': spatial_logits,
            'temporal_logits': temporal_logits,
            'features': features_sequence
        }
    
    def forward(
        self,
        frame_sequence: torch.Tensor,
        audio_sentiment: Optional[torch.Tensor] = None,
        audio_emotions: Optional[torch.Tensor] = None,
        audio_waveform: Optional[torch.Tensor] = None,
        landmarks_sequence: Optional[torch.Tensor] = None,
        person_id: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Full multimodal forward pass.
        
        Args:
            frame_sequence: Video frames (B, T, 3, H, W)
            audio_sentiment: Sentiment scores (B, T_a, 3) [positive, negative, neutral]
            audio_emotions: Emotion keyword scores (B, T_a, 7)
            audio_waveform: Raw audio waveform (B, T_w, samples) for acoustic analysis
            landmarks_sequence: Facial landmarks (B, T, 468*3) if use_mediapipe=True
            person_id: Person identifier for personalized acoustic profiling
            
        Returns:
            Dictionary with logits and intermediate predictions
        """
        B, T, C, H, W = frame_sequence.shape
        
        # Extract visual features
        features_list = []
        spatial_logits_list = []
        
        for t in range(T):
            frame = frame_sequence[:, t, :, :, :]
            features = self.spatial_cnn.extract_features(frame)
            features_list.append(features)
            spatial_logits = self.spatial_cnn.classifier(features)
            spatial_logits_list.append(spatial_logits)
        
        features_sequence = torch.stack(features_list, dim=1)  # (B, T, visual_dim)
        spatial_logits = torch.stack(spatial_logits_list, dim=1).mean(dim=1)
        
        # Temporal transformer (visual-only)
        temporal_logits = self.temporal_transformer(features_sequence)
        
        # Collect predictions
        all_logits = [spatial_logits, temporal_logits]
        prediction_names = ['spatial', 'temporal']
        
        # MediaPipe landmarks
        if self.use_mediapipe and landmarks_sequence is not None:
            landmark_logits_list = []
            for t in range(T):
                landmarks = landmarks_sequence[:, t, :]
                landmark_features = self.landmark_extractor.landmark_encoder(landmarks)
                landmark_logits = self.landmark_classifier(landmark_features)
                landmark_logits_list.append(landmark_logits)
            landmark_logits = torch.stack(landmark_logits_list, dim=1).mean(dim=1)
            all_logits.append(landmark_logits)
            prediction_names.append('landmark')
        
        # Audio features
        if self.use_audio and audio_sentiment is not None and audio_emotions is not None:
            T_a = audio_sentiment.shape[1]
            
            # Encode each audio segment
            audio_features_list = []
            audio_logits_list = []
            
            for t in range(T_a):
                sent = audio_sentiment[:, t, :]
                emot = audio_emotions[:, t, :]
                audio_feat, audio_log = self.audio_encoder(sent, emot)
                audio_features_list.append(audio_feat)
        # Acoustic features (spectrogram/prosody for tone)
        if self.use_acoustic and audio_waveform is not None:
            T_w = audio_waveform.shape[1]
            
            # Process each audio segment
            acoustic_logits_list = []
            acoustic_features_list = []
            
            for t in range(T_w):
                waveform = audio_waveform[:, t, :]  # (B, samples)
                
                if self.use_personalized_acoustic and person_id is not None:
                    # Extract raw acoustic features for profile computation
                    import librosa
                    
                    # Process each sample in the batch
                    batch_deviations = []
                    for b in range(waveform.size(0)):
                        waveform_np = waveform[b].cpu().numpy() if waveform.is_cuda else waveform[b].numpy()
                        
                        # Compute acoustic features
                        pitch = float(np.mean(librosa.yin(waveform_np, fmin=80, fmax=400, sr=16000)))
                        energy = float(np.mean(np.abs(waveform_np)))
                        
                        # Get/create profile and compute deviations
                        profile = self.profile_manager.get_or_create_profile(person_id)
                        
                        # Initialize baseline if first time
                        if profile.baseline_pitch is None:
                            profile.baseline_pitch = pitch
                            profile.baseline_energy = energy
                            profile.baseline_speaking_rate = 150.0  # default
                        
                        # Compute deviations
                        raw_features = {
                            'pitch': pitch,
                            'energy': energy,
                            'speaking_rate': profile.baseline_speaking_rate  # use stored value
                        }
                        deviations = profile.compute_deviation(raw_features)
                        batch_deviations.append([
                            deviations['pitch'],
                            deviations['energy'],
                            deviations['speaking_rate']
                        ])
                        
                        # Update profile with new observation (continuous adaptation)
                        self.profile_manager.update_profile(person_id, raw_features)
                    
                    # Create batch deviations tensor (B, 3)
                    deviations_tensor = torch.FloatTensor(batch_deviations).to(waveform.device)
                    
                    # Personalized encoding
                    acoustic_feat, acoustic_log = self.acoustic_encoder(
                        waveform,
                        person_id=person_id,
                        deviations=deviations_tensor
                    )
                else:
                    # Standard acoustic encoding (no personalization)
                    acoustic_feat, acoustic_log = self.acoustic_encoder(waveform)
                
                acoustic_logits_list.append(acoustic_log)
                acoustic_features_list.append(acoustic_feat)
            
            acoustic_logits = torch.stack(acoustic_logits_list, dim=1).mean(dim=1)
            all_logits.append(acoustic_logits)
            prediction_names.append('personalized_acoustic' if self.use_personalized_acoustic else 'acoustic')
        
        # Multimodal fusion
        if self.fusion_method == 'weighted':
            weights = F.softmax(self.fusion_weights[:len(all_logits)], dim=0)
            combined_logits = sum(w * logit for w, logit in zip(weights, all_logits))
        elif self.fusion_method == 'concat':
            concat_logits = torch.cat(all_logits, dim=1)
            combined_logits = self.fusion_fc(concat_logits)
        elif self.fusion_method == 'attention':
            # Compute attention weights for each modality
            attention_scores = []
            for logit in all_logits:
                score = self.modality_attention(logit)
                attention_scores.append(score)
            
            attention_weights = F.softmax(torch.cat(attention_scores, dim=1), dim=1)
            combined_logits = sum(attention_weights[:, i:i+1] * all_logits[i] 
                                for i in range(len(all_logits)))
        else:
            # Simple average
            combined_logits = sum(all_logits) / len(all_logits)
        
        # Build result dictionary
        result = {
            'logits': combined_logits,
            'probabilities': F.softmax(combined_logits, dim=1)
        }
        
        # Add individual predictions
        for name, logit in zip(prediction_names, all_logits):
            result[f'{name}_logits'] = logit
            result[f'{name}_probabilities'] = F.softmax(logit, dim=1)
        
        if self.fusion_method == 'weighted':
            result['fusion_weights'] = F.softmax(self.fusion_weights[:len(all_logits)], dim=0)
        
        return result
    
    def predict_emotion(
        self,
        frame_sequence: torch.Tensor,
        audio_sentiment: Optional[torch.Tensor] = None,
        audio_emotions: Optional[torch.Tensor] = None,
        audio_waveform: Optional[torch.Tensor] = None,
        landmarks_sequence: Optional[torch.Tensor] = None,
        person_id: Optional[str] = None
    ) -> Dict:
        """
        High-level prediction interface.
        
        Args:
            frame_sequence: Video frames (B, T, 3, H, W)
            audio_sentiment: Sentiment scores (B, T_a, 3)
            audio_emotions: Emotion keyword scores (B, T_a, 7)
            audio_waveform: Raw audio waveform (B, T_w, samples)
            landmarks_sequence: Facial landmarks (B, T, 468*3)
            person_id: Person identifier for personalized acoustic profiling
            
        Returns:
            Dictionary with predictions
        """
        with torch.no_grad():
            result = self.forward(
                frame_sequence, 
                audio_sentiment, 
                audio_emotions,
                audio_waveform,
                landmarks_sequence,
                person_id
            )
            
            probs = result['probabilities'][0].cpu().numpy()
            emotion_idx = np.argmax(probs)
            
            return {
                'emotion': EMOTION_LABELS[emotion_idx],
                'confidence': float(probs[emotion_idx]),
                'probabilities': {EMOTION_LABELS[i]: float(probs[i]) for i in range(self.num_emotions)},
                'all_predictions': {
                    name.replace('_probabilities', ''): {
                        EMOTION_LABELS[i]: float(result[name][0, i].cpu().item())
                        for i in range(self.num_emotions)
                    }
                    for name in result.keys() if name.endswith('_probabilities') and name != 'probabilities'
                }
            }
        
        return result


def test_multimodal_model():
    """Test the multimodal emotion recognizer."""
    print("=" * 70)
    print("Testing Multimodal Emotion Recognizer (Visual + Temporal + Audio)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create model
    model = MultimodalEmotionRecognizer(
        num_emotions=7,
        cnn_backbone='mobilenet_v2',
        pretrained=True,
        temporal_layers=2,
        temporal_heads=4,
        use_mediapipe=False,
        use_audio=True,
        fusion_method='weighted',
        dropout=0.3
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {num_params:,}")
    print(f"Modalities: Visual + Temporal + Audio")
    print(f"Fusion: weighted\n")
    
    # Test input
    batch_size = 2
    num_frames = 16
    num_audio_segments = 4
    
    frames = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    audio_sentiment = torch.randn(batch_size, num_audio_segments, 3).softmax(dim=-1).to(device)
    audio_emotions = torch.randn(batch_size, num_audio_segments, 7).softmax(dim=-1).to(device)
    
    print(f"Input shapes:")
    print(f"  Frames: {frames.shape}")
    print(f"  Audio sentiment: {audio_sentiment.shape}")
    print(f"  Audio emotions: {audio_emotions.shape}\n")
    
    # Forward pass
    result = model(frames, audio_sentiment, audio_emotions)
    
    print("Output:")
    print(f"  Combined logits: {result['logits'].shape}")
    print(f"  Combined probabilities: {result['probabilities'].shape}")
    
    if 'fusion_weights' in result:
        weights = result['fusion_weights'].cpu().detach().numpy()
        print(f"\nFusion weights:")
        modalities = ['spatial', 'temporal', 'audio', 'audiovisual']
        for i, (mod, w) in enumerate(zip(modalities, weights)):
            print(f"  {mod}: {w:.3f}")
    
    # Test prediction
    print("\nTesting prediction:")
    pred = model.predict_emotion(frames, audio_sentiment, audio_emotions)
    print(f"  Predicted emotion: {pred['emotion']}")
    print(f"  Confidence: {pred['confidence']:.3f}")
    print(f"  Top 3 emotions:")
    sorted_probs = sorted(pred['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for emotion, prob in sorted_probs[:3]:
        print(f"    {emotion}: {prob:.3f}")
    
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_multimodal_model()

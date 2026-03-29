# Audio Sentiment Analysis & Multimodal Fusion

This module provides audio sentiment analysis using speech-to-text and NLP, plus multimodal fusion combining audio and visual emotion detection.

## 🎤 Features

### Audio Sentiment Analysis
- **Speech-to-Text**: Converts speech to text using Google Speech Recognition or CMU Sphinx
- **Sentiment Analysis**: Uses transformer models (DistilBERT) or TextBlob for sentiment classification
- **Emotion Detection**: Keyword-based emotion detection from transcribed text
- **Phrase Detection**: Identifies positive and negative phrasing patterns
- **Real-time Analysis**: Continuous audio recording and analysis

### Multimodal Fusion
- **Visual + Audio**: Combines facial emotion detection with audio sentiment
- **Intelligent Fusion**: Weighted, average, or maximum fusion methods
- **Real-time**: Simultaneous video and audio processing
- **Comprehensive Output**: Frame-by-frame multimodal analysis

## 📁 Files

```
tools/audio/
├── audio_sentiment_analyzer.py    # Audio-only sentiment analysis
├── multimodal_fusion.py           # Visual + Audio fusion
└── README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core audio dependencies
pip install SpeechRecognition pyaudio

# NLP dependencies
pip install transformers textblob

# Download TextBlob corpora
python -m textblob.download_corpora

# Optional: For offline speech recognition
pip install pocketsphinx
```

### 2. Audio-Only Sentiment Analysis

```bash
# Real-time audio sentiment analysis
python tools/audio/audio_sentiment_analyzer.py

# With options
python tools/audio/audio_sentiment_analyzer.py \
    --duration 60 \
    --output-dir my_analysis \
    --chunk-duration 5 \
    --use-textblob
```

**Output:**
- Console: Real-time transcription and sentiment
- JSON file: Complete analysis with timestamps

### 3. Multimodal Fusion (Visual + Audio)

```bash
# Run multimodal analysis
python tools/audio/multimodal_fusion.py

# With GPU and trained model
python tools/audio/multimodal_fusion.py \
    --model-path checkpoints/best_model.pth \
    --use-mediapipe \
    --device cuda \
    --fusion weighted
```

**Features:**
- Visual emotion from facial expressions
- Audio sentiment from speech
- Fused emotion combining both modalities
- Real-time display with emotion bars

## 📊 How It Works

### Audio Sentiment Pipeline

```
Microphone Input
    ↓
Speech Recognition (Google/Sphinx)
    ↓
Text Transcription
    ↓
Sentiment Analysis (Transformer/TextBlob)
    ↓
Emotion Keyword Detection
    ↓
Positive/Negative Phrase Detection
    ↓
Analysis Output
```

### Multimodal Fusion

```
Webcam → Face Detection → Visual Emotion (CNN)
                                ↓
                              FUSION ← Audio Sentiment (NLP)
                                ↓                ↑
                          Fused Emotion    Microphone → Speech-to-Text
```

**Fusion Methods:**

1. **Weighted** (default): 60% visual + 40% audio compatibility
2. **Average**: Simple average of visual and audio signals
3. **Maximum**: Takes highest confidence from either modality

## 🎯 Detected Emotions & Sentiments

### Visual Emotions (7 classes)
- Happy
- Sad
- Angry
- Fear
- Surprise
- Disgust
- Neutral

### Audio Sentiments
- **Positive**: Happy, joyful expressions
- **Negative**: Sad, angry, frustrated expressions
- **Neutral**: Calm, matter-of-fact speech

### Sentiment-to-Emotion Mapping

The system maps audio sentiment to visual emotion compatibility:

| Audio Sentiment | Compatible Visual Emotions |
|----------------|----------------------------|
| Positive       | Happy (0.8), Surprise (0.5), Neutral (0.3) |
| Negative       | Sad (0.7), Angry (0.7), Disgust (0.6), Fear (0.5) |
| Neutral        | Neutral (0.8), all others (0.2-0.3) |

## 📝 Positive/Negative Phrase Detection

### Positive Phrases
- "I love", "I like", "I enjoy"
- "I appreciate", "I'm happy"
- "Feel good", "Feel great"
- "Thank you", "That's wonderful"
- "How nice", "I'm glad"

### Negative Phrases
- "I hate", "I don't like", "I dislike"
- "I'm upset", "I'm sad", "I'm angry"
- "Feel bad", "Feel terrible"
- "That's awful", "That's terrible"
- "How awful", "I'm disappointed"

## 💡 Usage Examples

### Example 1: Audio Sentiment Analysis

```bash
# Start recording
python tools/audio/audio_sentiment_analyzer.py

# Speak naturally:
# "I'm so happy today! Everything is going great."
# → Sentiment: POSITIVE (0.95)
# → Emotion: happy (0.85)
# → Phrases: ['i\'m happy', 'going great']

# "I'm really upset about this situation."
# → Sentiment: NEGATIVE (0.89)
# → Emotion: sad (0.60), angry (0.30)
# → Phrases: ['i\'m upset']
```

### Example 2: Multimodal Analysis

```bash
# Run multimodal fusion
python tools/audio/multimodal_fusion.py --device cuda

# Results show:
# Visual: happy (from smiling face)
# Audio: positive (from cheerful speech)
# Fused: happy (high confidence)

# Or detect mismatch:
# Visual: neutral (poker face)
# Audio: negative (complaining speech)
# Fused: sad/angry (audio dominates when visual is neutral)
```

### Example 3: Custom Duration

```bash
# Record for 2 minutes
python tools/audio/audio_sentiment_analyzer.py --duration 120

# Unlimited recording (Ctrl+C to stop)
python tools/audio/audio_sentiment_analyzer.py
```

## 📈 Output Format

### Audio Analysis JSON

```json
{
  "transcripts": [
    "I'm so happy today",
    "Everything is going great"
  ],
  "sentiments": [
    {
      "timestamp": 2.5,
      "text": "I'm so happy today",
      "sentiment": {
        "label": "POSITIVE",
        "score": 0.95,
        "sentiment": "positive",
        "confidence": 0.95
      },
      "emotion_scores": {
        "happy": 0.85,
        "neutral": 0.10,
        "sad": 0.05
      },
      "phrases": {
        "positive": ["i'm happy"],
        "negative": []
      }
    }
  ],
  "summary": {
    "total_segments": 10,
    "sentiment_distribution": {
      "positive": 7,
      "negative": 2,
      "neutral": 1
    },
    "dominant_sentiment": "positive",
    "positive_phrases_detected": 5,
    "negative_phrases_detected": 1
  }
}
```

### Multimodal Fusion JSON

```json
{
  "fusion_history": [
    {
      "frame": 100,
      "timestamp": 1638000000,
      "emotion": "happy",
      "confidence": 0.92,
      "visual_emotion": "happy",
      "audio_sentiment": "positive",
      "audio_text": "I'm so happy today",
      "audio_confidence": 0.95,
      "fusion_method": "weighted"
    }
  ],
  "summary": {
    "total_frames": 500,
    "emotion_distribution": {
      "happy": 350,
      "neutral": 100,
      "sad": 50
    },
    "dominant_emotion": "happy"
  }
}
```

## 🔧 Configuration

### Audio Sentiment Analyzer Options

```bash
--output-dir DIR          # Output directory (default: audio_analysis)
--duration SECONDS        # Recording duration (default: unlimited)
--chunk-duration SECONDS  # Audio chunk size (default: 5)
--use-textblob           # Use TextBlob instead of transformer
--engine ENGINE          # Speech engine: google, sphinx (default: google)
```

### Multimodal Fusion Options

```bash
--model-path PATH        # Path to visual emotion model
--use-mediapipe         # Use MediaPipe-enhanced visual model
--device DEVICE         # cpu or cuda
--output-dir DIR        # Output directory
--fusion METHOD         # weighted, average, or max
```

## 🎓 Use Cases

### 1. Customer Service Analysis
- Detect customer satisfaction from voice and facial expressions
- Identify frustration or confusion in real-time
- Measure sentiment throughout conversation

### 2. Mental Health Monitoring
- Track emotional states over time
- Detect incongruence between speech and facial expressions
- Identify stress or anxiety patterns

### 3. Video Conferencing
- Real-time emotion awareness
- Meeting sentiment analysis
- Participant engagement tracking

### 4. Content Analysis
- Analyze video content for emotional impact
- Study speaker authenticity (visual-audio alignment)
- Research emotion expression patterns

### 5. Education & Training
- Monitor student engagement
- Assess comprehension through emotion cues
- Provide feedback on presentation skills

## 🔍 Advanced Features

### Emotion Keyword Detection

The system detects emotions from specific keywords:

```python
emotion_keywords = {
    'happy': ['happy', 'joy', 'excited', 'wonderful', 'love', 'amazing'],
    'sad': ['sad', 'unhappy', 'depressed', 'disappointed', 'upset'],
    'angry': ['angry', 'mad', 'furious', 'irritated', 'frustrated'],
    'fear': ['afraid', 'scared', 'frightened', 'worried', 'anxious'],
    'surprise': ['surprised', 'shocked', 'amazed', 'astonished'],
    'neutral': ['okay', 'fine', 'alright', 'normal']
}
```

### Sentiment Analysis Models

**Transformer (DistilBERT):**
- Pre-trained on Stanford Sentiment Treebank
- High accuracy (>90%)
- Requires ~500MB model download
- GPU recommended for speed

**TextBlob (Fallback):**
- Lightweight, no model download
- Based on pattern analysis
- Good for basic sentiment
- Works offline

## 🐛 Troubleshooting

### No Audio Detected
```bash
# Check microphone
python -c "import pyaudio; p = pyaudio.PyAudio(); print(p.get_default_input_device_info())"

# Adjust energy threshold
# In code: recognizer.energy_threshold = 300  # Lower = more sensitive
```

### Speech Recognition Fails
```bash
# Try offline recognition
pip install pocketsphinx
python tools/audio/audio_sentiment_analyzer.py --engine sphinx

# Or check internet connection for Google API
```

### Transformer Model Errors
```bash
# Use TextBlob instead
python tools/audio/audio_sentiment_analyzer.py --use-textblob

# Or reinstall transformers
pip install --upgrade transformers torch
```

### Low FPS in Multimodal
```bash
# Use GPU
python tools/audio/multimodal_fusion.py --device cuda

# Increase chunk duration (less frequent audio processing)
# Modify: chunk_duration=10 in code
```

## 📊 Performance

### Audio Analysis
- **Transcription**: ~1-2s delay per 5s chunk
- **Sentiment Analysis**: <100ms per text
- **Memory**: ~200MB (TextBlob), ~700MB (Transformer)

### Multimodal Fusion
- **FPS**: 20-30 (CPU), 40-60 (GPU)
- **Audio Latency**: ~1-3s
- **Memory**: ~1GB (CPU), ~2.5GB (GPU)

## 📚 API Usage

### Python API Example

```python
from tools.audio.audio_sentiment_analyzer import AudioSentimentAnalyzer

# Create analyzer
analyzer = AudioSentimentAnalyzer(
    use_transformer=True,
    output_dir='my_analysis'
)

# Analyze text directly
text = "I'm so happy today!"
result = analyzer.analyze_sentiment_transformer(text)
print(result)
# {'label': 'POSITIVE', 'score': 0.95, 'sentiment': 'positive'}

# Detect phrases
phrases = analyzer.detect_positive_negative_phrases(text)
print(phrases)
# {'positive': ['i\'m happy'], 'negative': []}

# Run real-time analysis
analyzer.run_realtime(duration=60)
```

### Multimodal API Example

```python
from tools.audio.multimodal_fusion import MultimodalEmotionFusion

# Create fusion system
fusion = MultimodalEmotionFusion(
    model_path='checkpoints/best_model.pth',
    use_mediapipe_model=True,
    device='cuda',
    fusion_method='weighted'
)

# Run analysis
fusion.run()

# Access results
print(fusion.generate_summary())
```

## 🔗 Integration

### With Video Recorder

```python
# Record video with audio sentiment overlay
# (Future enhancement)
```

### With Training Pipeline

```python
# Use audio sentiment as additional training signal
# (Future enhancement)
```

## 📖 References

- **Speech Recognition**: Google Speech API, CMU Sphinx
- **Sentiment Analysis**: DistilBERT (HuggingFace)
- **TextBlob**: Pattern-based NLP
- **Visual Emotions**: See main project README

## 🤝 Contributing

To add new emotion keywords or phrases:

1. Edit `emotion_keywords` dictionary in `audio_sentiment_analyzer.py`
2. Edit `positive_phrases` and `negative_phrases` lists
3. Test with diverse speech samples

## 📝 License

Same as main project.

---

**Quick Start**: `python tools/audio/audio_sentiment_analyzer.py`

"""
Audio Sentiment Analysis with Speech-to-Text
Records audio, transcribes speech, and analyzes sentiment/emotion from text.
Combines with visual emotion detection for multimodal analysis.
"""

import sys
from pathlib import Path
# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pyaudio
import wave
import threading
import queue
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
import os
import numpy as np

try:
    import speech_recognition as sr
except ImportError:
    print("Warning: speech_recognition not installed. Run: pip install SpeechRecognition")
    sr = None

try:
    from transformers import pipeline
except ImportError:
    print("Warning: transformers not installed. Run: pip install transformers")
    pipeline = None

try:
    from textblob import TextBlob
except ImportError:
    print("Warning: textblob not installed. Run: pip install textblob")
    TextBlob = None


class AudioSentimentAnalyzer:
    """
    Records audio, transcribes speech, and analyzes sentiment.
    """
    def __init__(
        self,
        use_transformer: bool = True,
        output_dir: str = 'audio_analysis',
        sample_rate: int = 16000,
        chunk_duration: int = 5,
        recognition_engine: str = 'google'
    ):
        """
        Args:
            use_transformer: Use transformer-based sentiment analysis (more accurate)
            output_dir: Directory to save audio recordings and analysis
            sample_rate: Audio sample rate in Hz
            chunk_duration: Duration of audio chunks for processing (seconds)
            recognition_engine: Speech recognition engine ('google', 'sphinx', 'whisper')
        """
        self.output_dir = output_dir
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.recognition_engine = recognition_engine
        self.use_transformer = use_transformer and pipeline is not None
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize speech recognizer
        if sr is None:
            raise ImportError("SpeechRecognition not installed. Run: pip install SpeechRecognition")
        
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
        # Initialize sentiment analyzer
        if self.use_transformer:
            print("Loading transformer sentiment model...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            print("✓ Transformer model loaded")
        else:
            self.sentiment_analyzer = None
            print("Using TextBlob for sentiment analysis")
        
        # Emotion keywords for detailed analysis
        self.emotion_keywords = {
            'happy': ['happy', 'joy', 'excited', 'wonderful', 'great', 'love', 'amazing', 
                     'fantastic', 'excellent', 'delighted', 'cheerful', 'pleased'],
            'sad': ['sad', 'unhappy', 'depressed', 'down', 'miserable', 'gloomy', 
                   'disappointed', 'upset', 'hurt', 'heartbroken'],
            'angry': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated',
                     'rage', 'hate', 'disgusted', 'outraged'],
            'fear': ['afraid', 'scared', 'frightened', 'worried', 'anxious', 'nervous',
                    'terrified', 'panic', 'fear', 'concerned'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected',
                        'stunned', 'wow', 'incredible'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'regular', 'usual']
        }
        
        # Positive/Negative phrases
        self.positive_phrases = [
            'i love', 'i like', 'i enjoy', 'i appreciate', 'i\'m happy',
            'feel good', 'feel great', 'thank you', 'that\'s wonderful',
            'that\'s great', 'how nice', 'how wonderful', 'i\'m glad'
        ]
        
        self.negative_phrases = [
            'i hate', 'i don\'t like', 'i dislike', 'i\'m upset', 'i\'m sad',
            'feel bad', 'feel terrible', 'i\'m angry', 'that\'s awful',
            'that\'s terrible', 'how awful', 'i\'m disappointed'
        ]
        
        # Recording state
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.transcript_history = []
        self.sentiment_history = []
        
    def detect_emotion_from_text(self, text: str) -> Dict[str, float]:
        """
        Detect emotion from text using keyword matching.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with emotion scores
        """
        text_lower = text.lower()
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
        
        # Count keyword matches
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotion_scores[emotion] += 1
        
        # Normalize scores
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: v/total for k, v in emotion_scores.items()}
        else:
            emotion_scores['neutral'] = 1.0
        
        return emotion_scores
    
    def detect_positive_negative_phrases(self, text: str) -> Dict[str, List[str]]:
        """
        Detect positive and negative phrases in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with 'positive' and 'negative' phrase lists
        """
        text_lower = text.lower()
        detected = {'positive': [], 'negative': []}
        
        for phrase in self.positive_phrases:
            if phrase in text_lower:
                detected['positive'].append(phrase)
        
        for phrase in self.negative_phrases:
            if phrase in text_lower:
                detected['negative'].append(phrase)
        
        return detected
    
    def analyze_sentiment_transformer(self, text: str) -> Dict:
        """
        Analyze sentiment using transformer model.
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        if not text.strip():
            return {
                'label': 'NEUTRAL',
                'score': 0.5,
                'sentiment': 'neutral'
            }
        
        try:
            result = self.sentiment_analyzer(text)[0]
            sentiment = 'positive' if result['label'] == 'POSITIVE' else 'negative'
            
            return {
                'label': result['label'],
                'score': result['score'],
                'sentiment': sentiment,
                'confidence': result['score']
            }
        except Exception as e:
            print(f"Transformer analysis failed: {e}")
            return self.analyze_sentiment_textblob(text)
    
    def analyze_sentiment_textblob(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob (fallback method).
        
        Args:
            text: Input text
            
        Returns:
            Sentiment analysis results
        """
        if TextBlob is None or not text.strip():
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment': 'neutral',
                'confidence': 0.5
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            # Convert polarity to confidence score
            confidence = abs(polarity)
            
            return {
                'polarity': polarity,
                'subjectivity': subjectivity,
                'sentiment': sentiment,
                'confidence': confidence
            }
        except Exception as e:
            print(f"TextBlob analysis failed: {e}")
            return {
                'polarity': 0.0,
                'subjectivity': 0.5,
                'sentiment': 'neutral',
                'confidence': 0.5
            }
    
    def transcribe_audio(self, audio_data: sr.AudioData) -> Optional[str]:
        """
        Transcribe audio to text using speech recognition.
        
        Args:
            audio_data: Audio data from microphone
            
        Returns:
            Transcribed text or None
        """
        try:
            if self.recognition_engine == 'google':
                text = self.recognizer.recognize_google(audio_data)
            elif self.recognition_engine == 'sphinx':
                text = self.recognizer.recognize_sphinx(audio_data)
            else:
                text = self.recognizer.recognize_google(audio_data)
            
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            return None
    
    def analyze_audio_chunk(self, audio_data: sr.AudioData, timestamp: float) -> Dict:
        """
        Transcribe and analyze an audio chunk.
        
        Args:
            audio_data: Audio data
            timestamp: Timestamp of recording
            
        Returns:
            Analysis results
        """
        # Transcribe
        text = self.transcribe_audio(audio_data)
        
        if text is None:
            return {
                'timestamp': timestamp,
                'text': None,
                'sentiment': None,
                'emotion_scores': None,
                'phrases': None
            }
        
        # Analyze sentiment
        if self.use_transformer:
            sentiment_result = self.analyze_sentiment_transformer(text)
        else:
            sentiment_result = self.analyze_sentiment_textblob(text)
        
        # Detect emotions from keywords
        emotion_scores = self.detect_emotion_from_text(text)
        
        # Detect positive/negative phrases
        phrases = self.detect_positive_negative_phrases(text)
        
        # Combine results
        result = {
            'timestamp': timestamp,
            'text': text,
            'sentiment': sentiment_result,
            'emotion_scores': emotion_scores,
            'phrases': phrases,
            'word_count': len(text.split())
        }
        
        return result
    
    def record_audio_chunk(self, microphone: sr.Microphone) -> Optional[sr.AudioData]:
        """
        Record a chunk of audio from microphone.
        
        Args:
            microphone: Microphone instance
            
        Returns:
            Audio data or None
        """
        try:
            with microphone as source:
                # Adjust for ambient noise
                if len(self.transcript_history) == 0:
                    print("Adjusting for ambient noise... Please wait.")
                    self.recognizer.adjust_for_ambient_noise(source, duration=1)
                    print("✓ Ready to record")
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=self.chunk_duration, phrase_time_limit=self.chunk_duration)
                return audio
        except Exception as e:
            print(f"Recording error: {e}")
            return None
    
    def save_analysis(self, output_path: str):
        """
        Save analysis results to JSON file.
        
        Args:
            output_path: Path to save JSON file
        """
        data = {
            'transcripts': self.transcript_history,
            'sentiments': self.sentiment_history,
            'summary': self.generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, indent=2, fp=f)
        
        print(f"✓ Analysis saved to: {output_path}")
    
    def generate_summary(self) -> Dict:
        """
        Generate summary statistics from analysis.
        
        Returns:
            Summary dictionary
        """
        if not self.sentiment_history:
            return {}
        
        # Count sentiments
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0
        total_phrases = {'positive': [], 'negative': []}
        all_emotions = {emotion: [] for emotion in self.emotion_keywords.keys()}
        
        for item in self.sentiment_history:
            if item['sentiment']:
                sentiment = item['sentiment'].get('sentiment', 'neutral')
                sentiment_counts[sentiment] += 1
                total_confidence += item['sentiment'].get('confidence', 0)
            
            if item['phrases']:
                total_phrases['positive'].extend(item['phrases']['positive'])
                total_phrases['negative'].extend(item['phrases']['negative'])
            
            if item['emotion_scores']:
                for emotion, score in item['emotion_scores'].items():
                    if score > 0:
                        all_emotions[emotion].append(score)
        
        # Calculate averages
        total = len(self.sentiment_history)
        avg_confidence = total_confidence / total if total > 0 else 0
        
        # Dominant emotion
        avg_emotions = {k: np.mean(v) if v else 0 for k, v in all_emotions.items()}
        dominant_emotion = max(avg_emotions.items(), key=lambda x: x[1])[0] if avg_emotions else 'neutral'
        
        return {
            'total_segments': total,
            'sentiment_distribution': sentiment_counts,
            'average_confidence': avg_confidence,
            'dominant_sentiment': max(sentiment_counts.items(), key=lambda x: x[1])[0],
            'positive_phrases_detected': len(set(total_phrases['positive'])),
            'negative_phrases_detected': len(set(total_phrases['negative'])),
            'dominant_emotion': dominant_emotion,
            'emotion_averages': avg_emotions
        }
    
    def print_analysis(self, result: Dict):
        """
        Print analysis results to console.
        
        Args:
            result: Analysis result dictionary
        """
        if result['text'] is None:
            print("  [No speech detected]")
            return
        
        print(f"\n{'='*70}")
        print(f"Text: {result['text']}")
        print(f"{'='*70}")
        
        # Sentiment
        if result['sentiment']:
            sentiment = result['sentiment']
            if 'label' in sentiment:
                print(f"Sentiment: {sentiment['label']} (confidence: {sentiment['score']:.3f})")
            else:
                print(f"Sentiment: {sentiment['sentiment'].upper()} "
                      f"(polarity: {sentiment.get('polarity', 0):.3f})")
        
        # Emotions
        if result['emotion_scores']:
            top_emotions = sorted(result['emotion_scores'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            print(f"Emotions: ", end="")
            print(", ".join([f"{e}: {s:.2f}" for e, s in top_emotions if s > 0]))
        
        # Phrases
        if result['phrases']:
            if result['phrases']['positive']:
                print(f"✓ Positive phrases: {', '.join(result['phrases']['positive'])}")
            if result['phrases']['negative']:
                print(f"✗ Negative phrases: {', '.join(result['phrases']['negative'])}")
        
        print()
    
    def run_realtime(self, duration: Optional[int] = None):
        """
        Run real-time audio recording and analysis.
        
        Args:
            duration: Duration in seconds (None for indefinite)
        """
        print("=" * 70)
        print("Real-time Audio Sentiment Analysis")
        print("=" * 70)
        print("Starting audio recording...")
        print("Speak naturally. Press Ctrl+C to stop.\n")
        
        microphone = sr.Microphone(sample_rate=self.sample_rate)
        start_time = time.time()
        segment_count = 0
        
        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Record audio chunk
                audio_data = self.record_audio_chunk(microphone)
                if audio_data is None:
                    continue
                
                timestamp = time.time() - start_time
                segment_count += 1
                
                print(f"\n--- Segment {segment_count} ({timestamp:.1f}s) ---")
                
                # Analyze
                result = self.analyze_audio_chunk(audio_data, timestamp)
                
                # Store results
                self.transcript_history.append(result['text'])
                self.sentiment_history.append(result)
                
                # Print results
                self.print_analysis(result)
        
        except KeyboardInterrupt:
            print("\n\nStopping recording...")
        
        finally:
            # Print summary
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            
            summary = self.generate_summary()
            print(f"Total segments: {summary.get('total_segments', 0)}")
            print(f"Dominant sentiment: {summary.get('dominant_sentiment', 'N/A').upper()}")
            print(f"Dominant emotion: {summary.get('dominant_emotion', 'N/A').upper()}")
            print(f"Positive phrases: {summary.get('positive_phrases_detected', 0)}")
            print(f"Negative phrases: {summary.get('negative_phrases_detected', 0)}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.output_dir, f"analysis_{timestamp}.json")
            self.save_analysis(output_file)
            
            print(f"\n✓ Analysis complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Real-time Audio Sentiment Analysis with Speech-to-Text'
    )
    parser.add_argument('--output-dir', type=str, default='audio_analysis',
                       help='Directory to save analysis results')
    parser.add_argument('--duration', type=int, default=None,
                       help='Recording duration in seconds (default: unlimited)')
    parser.add_argument('--chunk-duration', type=int, default=5,
                       help='Duration of audio chunks for processing')
    parser.add_argument('--use-textblob', action='store_true',
                       help='Use TextBlob instead of transformer model')
    parser.add_argument('--engine', type=str, default='google',
                       choices=['google', 'sphinx'],
                       help='Speech recognition engine')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = AudioSentimentAnalyzer(
        use_transformer=not args.use_textblob,
        output_dir=args.output_dir,
        chunk_duration=args.chunk_duration,
        recognition_engine=args.engine
    )
    
    # Run analysis
    analyzer.run_realtime(duration=args.duration)


if __name__ == "__main__":
    main()

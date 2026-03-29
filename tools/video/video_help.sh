#!/bin/bash
# Video Recording Quick Help
# Run this to see a quick overview of video recording features

cat << 'EOF'

╔══════════════════════════════════════════════════════════════════════╗
║         VIDEO RECORDING WITH FACIAL LANDMARKS & ATTENTION            ║
╔══════════════════════════════════════════════════════════════════════╗

📹 QUICK COMMANDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  RECORD FROM WEBCAM
    python video_recorder_with_landmarks.py

    Controls during recording:
    • R - Start/Stop Recording
    • L - Toggle Landmarks (468 facial points)
    • A - Toggle Attention Heatmap
    • S - Save Screenshot  
    • Q - Quit

2️⃣  PROCESS EXISTING VIDEO
    python process_video.py input_video.mp4
    
    Output:
    • input_video_processed.mp4 (video with overlays)
    • input_video_processed_analysis.json (frame data)

3️⃣  ANALYZE ATTENTION PATTERNS
    python analyze_attention.py video_analysis.json
    
    Creates attention heatmap visualization

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🚀 COMMON USAGE PATTERNS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Record with GPU acceleration:
  python video_recorder_with_landmarks.py --device cuda

Record with trained model:
  python video_recorder_with_landmarks.py \
    --model-path checkpoints/best_model.pth \
    --device cuda

Process video with MediaPipe model:
  python process_video.py input.mp4 \
    --output output.mp4 \
    --use-mediapipe \
    --device cuda

Batch process (no display):
  python process_video.py video.mp4 \
    --output processed.mp4 \
    --no-display \
    --device cuda

Compare emotions:
  python analyze_attention.py analysis.json \
    --compare happy sad

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✨ WHAT YOU GET
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📍 468 Facial Landmarks
   • Eyes, eyebrows, mouth, nose tracking
   • Smart highlighting based on emotion
   • Real-time visualization

🔥 Attention Heatmaps
   • Shows where CNN model focuses
   • Color-coded (red=high, blue=low)
   • Per-region attention scores

📊 Emotion Analysis
   • 7 emotions: happy, sad, angry, fear, surprise, disgust, neutral
   • Confidence scores
   • Frame-by-frame tracking
   • Statistics and distribution

📁 Comprehensive Data
   • MP4 video output
   • JSON analysis file
   • Attention scores per region
   • Emotion timeline

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎯 ATTENTION PATTERNS BY EMOTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Happy:     High attention on MOUTH (smile) + eyes
Sad:       Focus on MOUTH corners + eyebrows (downturned)
Angry:     Strong attention on EYEBROWS + eyes (narrowed)
Surprise:  EYES (wide) + EYEBROWS (raised)
Fear:      Similar to surprise + mouth emphasis
Disgust:   NOSE + upper lip area
Neutral:   Balanced across all features

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📖 DOCUMENTATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full Documentation:
  README_VIDEO_RECORDING.md    - Complete guide (545 lines)
  VIDEO_EXAMPLES.md            - Copy-paste examples (638 lines)
  VIDEO_IMPLEMENTATION_SUMMARY.md - Technical summary

Quick Reference:
  QUICK_REFERENCE.md           - Command cheatsheet

Main Documentation:
  README.md                    - Project overview

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

No face detected:
  ✓ Ensure good lighting
  ✓ Face should be frontal and centered
  ✓ Minimum face size: 100x100 pixels

Low FPS:
  ✓ Use GPU: --device cuda
  ✓ Disable attention: --no-attention
  ✓ Disable landmarks: --no-landmarks
  ✓ Lower FPS: --fps 15

Import errors:
  pip install opencv-python torch mediapipe matplotlib

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚡ PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Real-time Recording:
  CPU: 20-30 FPS (all features)
  GPU: 45-60 FPS (all features)

Video Processing:
  CPU: ~15-20 FPS
  GPU: ~40-50 FPS

Processing Time (1920x1080 @ 30fps):
  30 seconds → ~60s CPU, ~20s GPU
  2 minutes  → ~240s CPU, ~60s GPU
  5 minutes  → ~600s CPU, ~150s GPU

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💡 PRO TIPS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Always use GPU for faster processing (--device cuda)
2. Good lighting is crucial for landmark detection
3. Hold each emotion for 2-3 seconds when recording
4. Use --no-display for batch processing
5. Toggle features to find best performance/quality balance
6. Analyze attention to understand model behavior
7. Compare emotions to validate model focus

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🎓 USE CASES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✓ Model debugging and validation
✓ Research and data analysis  
✓ Teaching and demonstrations
✓ Dataset creation
✓ Understanding model decisions
✓ Video conferencing emotion analysis
✓ Mental health monitoring
✓ Interactive applications

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📝 EXAMPLE WORKFLOW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Step 1: Record a video
python video_recorder_with_landmarks.py --device cuda

# (In viewer: Press R to record, express emotions, R to stop, Q to quit)

# Step 2: Process and analyze (if needed)
python process_video.py recordings/recording_*.mp4 \
  --output analyzed.mp4 \
  --device cuda

# Step 3: Analyze attention patterns
python analyze_attention.py analyzed_analysis.json

# Step 4: Compare emotions
python analyze_attention.py analyzed_analysis.json \
  --compare happy sad

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Need more help? Check README_VIDEO_RECORDING.md for complete documentation!

╚══════════════════════════════════════════════════════════════════════╝

EOF

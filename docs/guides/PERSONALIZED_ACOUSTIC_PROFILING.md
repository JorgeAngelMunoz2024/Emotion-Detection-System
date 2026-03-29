# Personalized Acoustic Profiling

## Overview

The personalized acoustic profiling system learns **individual-specific vocal characteristics** through unsupervised adaptation. It recognizes that emotional expression varies significantly between people—what sounds "excited" for one person might be normal speaking tone for another.

---

## Why Personalization Matters

### Individual Differences in Vocal Expression

**Person A** (Naturally High Energy):
- Baseline pitch: 250 Hz
- Baseline energy: 0.8
- "Happy" pitch: 280 Hz (+12%)
- "Sad" pitch: 230 Hz (-8%)

**Person B** (Naturally Low Energy):
- Baseline pitch: 150 Hz
- Baseline energy: 0.3
- "Happy" pitch: 165 Hz (+10%)
- "Sad" pitch: 140 Hz (-7%)

**Without personalization**: Person A's neutral tone might be classified as "excited"  
**With personalization**: System learns each person's unique baseline and detects deviations

---

## System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                    AUDIO WAVEFORM INPUT                        │
│                    (Person Specific)                           │
└───────────────────┬───────────────────────────────────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  Feature Extraction │
         │  • Pitch            │
         │  • Energy           │
         │  • Speaking Rate    │
         │  • Spectral Content │
         └──────────┬──────────┘
                    │
        ┌───────────┴──────────┐
        │                      │
        ▼                      ▼
┌──────────────┐      ┌────────────────────┐
│ Person       │      │ Universal          │
│ Profile      │      │ Acoustic           │
│ Manager      │      │ Encoder            │
│              │      │ (Spectrogram CNN)  │
│ • Baseline   │      └─────────┬──────────┘
│ • History    │                │
│ • Statistics │                │
└──────┬───────┘                │
       │                        │
       │  Compute Deviations    │
       │  (z-scores)            │
       │                        │
       └────────┬───────────────┘
                │
                ▼
      ┌──────────────────────┐
      │  Person-Specific     │
      │  Deviation Encoder   │
      │                      │
      │  Δpitch, Δenergy... │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Feature Fusion      │
      │  (Base + Deviations) │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Person-Specific     │
      │  Adaptation Layers   │
      │  (Per-person NN)     │
      └──────────┬───────────┘
                 │
                 ▼
      ┌──────────────────────┐
      │  Personalized        │
      │  Emotion Classifier  │
      │                      │
      │  Output: Emotion     │
      └──────────────────────┘
```

---

## Key Components

### 1. Personalized Acoustic Profile

Stores person-specific vocal characteristics:

```python
class PersonalizedAcousticProfile:
    - person_id: Unique identifier
    - baseline_pitch: Average pitch (adapts over time)
    - baseline_energy: Average volume/intensity
    - baseline_speaking_rate: Typical speech speed
    - emotion_acoustics: Per-emotion statistics
    - num_samples: Adaptation confidence
```

**Adaptation**: Uses exponential moving average to continuously learn

```python
new_baseline = (1 - α) × old_baseline + α × new_observation

where α = 0.1 (adaptation rate)
```

### 2. Deviation-Based Normalization

Converts absolute features to **person-relative** deviations:

```python
pitch_deviation = (current_pitch - baseline_pitch) / baseline_pitch
energy_deviation = (current_energy - baseline_energy) / baseline_energy
```

**Example**:
- Person A: current_pitch=280Hz, baseline=250Hz → deviation = +12%
- Person B: current_pitch=165Hz, baseline=150Hz → deviation = +10%

Both show similar excitement despite different absolute pitches!

### 3. Person-Specific Adaptation Layers

Each person gets dedicated neural network layers:

```python
self.person_adaptation[person_id] = nn.Sequential(
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256)
)
```

These layers learn person-specific patterns through usage.

### 4. Emotion-Specific Distributions

For supervised adaptation, system builds per-emotion statistics:

```python
person.emotion_acoustics['happy'] = {
    'pitch': [265, 270, 275, 268, ...],    # Observed pitches when happy
    'energy': [0.82, 0.85, 0.88, ...],     # Observed energies
    'spectral': [...]                       # Spectral features
}
```

Can compute likelihood: *How typical is this acoustic pattern for this person being happy?*

---

## Usage Workflow

### 1. First-Time User (Cold Start)

```python
from models.personalized_acoustic_profiling import PersonalizedAcousticProfileManager

# Initialize manager
manager = PersonalizedAcousticProfileManager("./profiles")

# New user
person_id = "alice_001"

# Process first audio samples (builds initial baseline)
for i in range(10):  # Collect ~10 samples
    waveform = record_audio()  # Record 2-3 seconds
    features = extract_acoustic_features(waveform)
    manager.update_profile(person_id, features)

print("✓ Initial profile created")
```

### 2. Ongoing Adaptation (Continuous Learning)

```python
# Every time user speaks, update profile
waveform = record_audio()
features = extract_acoustic_features(waveform)

# Optional: if emotion is known (e.g., from self-report or other modalities)
emotion = "happy"  # or None if unknown
manager.update_profile(person_id, features, emotion=emotion)

# Profile automatically adapts over time
```

### 3. Personalized Inference

```python
from models.personalized_acoustic_profiling import PersonalizedAcousticEncoder

# Create encoder
encoder = PersonalizedAcousticEncoder()
encoder.add_person(person_id)

# Get person-specific deviations
waveform = record_audio()
features = extract_acoustic_features(waveform)
deviations = manager.get_personalized_features(person_id, features)

# Personalized emotion prediction
with torch.no_grad():
    waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)
    features, logits = encoder(
        waveform_tensor, 
        person_id=person_id,
        deviations=deviations.unsqueeze(0)
    )
    
emotion = EMOTION_LABELS[logits.argmax()]
print(f"Detected emotion: {emotion}")
```

### 4. Profile Management

```python
# List all profiles
profiles = manager.list_profiles()
print(f"Registered users: {profiles}")

# Get profile statistics
stats = manager.get_profile_stats(person_id)
print(f"Samples collected: {stats['num_samples']}")
print(f"Baseline pitch: {stats['baseline_pitch']:.1f} Hz")
print(f"Emotions sampled: {stats['emotions_sampled']}")

# Save profiles to disk
manager.save_all_profiles()

# Load existing profile
profile = PersonalizedAcousticProfile.load(f"profiles/{person_id}.pkl")
```

---

## Integration with Multimodal System

```python
from models.audio_emotion_fusion import MultimodalEmotionRecognizer
from models.personalized_acoustic_profiling import PersonalizedAcousticProfileManager

# Create multimodal model with personalized acoustics
model = MultimodalEmotionRecognizer(
    use_acoustic=True,  # Enable acoustic modality
    use_audio=True,      # Enable NLP modality
    use_mediapipe=True   # Enable landmarks
)

# Profile manager
manager = PersonalizedAcousticProfileManager()

# Inference with personalization
def predict_emotion(person_id, video_frames, audio_waveform, audio_nlp_features):
    # Extract acoustic features
    acoustic_features = extract_acoustic_features(audio_waveform)
    
    # Get person-specific deviations
    deviations = manager.get_personalized_features(person_id, acoustic_features)
    
    # Forward pass with all modalities
    result = model(
        video_frames,
        audio_nlp_features['sentiment'],
        audio_nlp_features['emotions'],
        audio_waveform,  # Personalized acoustic encoder uses this
        landmarks=None
    )
    
    # Update profile (continuous adaptation)
    predicted_emotion = EMOTION_LABELS[result['logits'].argmax()]
    manager.update_profile(person_id, acoustic_features, emotion=predicted_emotion)
    
    return result
```

---

## Benefits

### 1. **Improved Accuracy**
- Accounts for individual baseline differences
- Reduces false positives from vocal variety

### 2. **Fairness**
- Doesn't penalize naturally loud/quiet speakers
- Works equally well across vocal ranges

### 3. **Privacy-Preserving**
- Profiles stored locally
- No centralized vocal database needed

### 4. **Continuous Improvement**
- Adapts to voice changes (fatigue, aging, mood)
- Learns from every interaction

### 5. **Unsupervised**
- Works without labeled emotion data
- Learns patterns naturally from usage

---

## Example: Two Users

### User A (High-Energy Speaker)

**Baseline Profile** (after 100 samples):
```python
{
    'baseline_pitch': 245 Hz,
    'baseline_energy': 0.75,
    'baseline_speaking_rate': 180 wpm
}
```

**Sample Utterances**:
1. "I'm so excited!" → pitch=290Hz (+18%) → **happy** ✓
2. "This is terrible." → pitch=220Hz (-10%) → **sad** ✓
3. "Let's begin." → pitch=245Hz (0%) → **neutral** ✓

### User B (Low-Energy Speaker)

**Baseline Profile** (after 100 samples):
```python
{
    'baseline_pitch': 155 Hz,
    'baseline_energy': 0.35,
    'baseline_speaking_rate': 120 wpm
}
```

**Sample Utterances**:
1. "I'm so excited!" → pitch=175Hz (+13%) → **happy** ✓
2. "This is terrible." → pitch=145Hz (-6%) → **sad** ✓
3. "Let's begin." → pitch=155Hz (0%) → **neutral** ✓

**Without Personalization**:
- User B's excited voice might be classified as neutral (absolute pitch too low)
- User A's neutral voice might be classified as excited (absolute pitch too high)

**With Personalization**:
- Both users' emotions correctly detected by measuring **deviation from their individual baselines**

---

## Advanced Features

### 1. Emotion-Specific Modeling

Build statistical models for each emotion:

```python
# After collecting labeled samples
happy_stats = profile.get_emotion_statistics('happy')
# Returns: {'pitch': (mean=275, std=15), 'energy': (mean=0.85, std=0.08)}

# Compute likelihood during inference
likelihood = gaussian_pdf(current_pitch, happy_stats['pitch'][0], happy_stats['pitch'][1])
```

### 2. Confidence Scoring

```python
if profile.num_samples < 10:
    confidence = 'LOW'  # Cold start
elif profile.num_samples < 50:
    confidence = 'MEDIUM'  # Learning
else:
    confidence = 'HIGH'  # Well-calibrated
```

### 3. Multi-Person Support

```python
# Family/household scenario
family_members = ['alice', 'bob', 'charlie']

for member in family_members:
    encoder.add_person(member)
    manager.get_or_create_profile(member)

# Automatic speaker identification (future work)
# detected_person = identify_speaker(waveform)
# result = encoder(waveform, person_id=detected_person)
```

### 4. Transfer Learning

```python
# Bootstrap new user from similar demographics
new_profile = PersonalizedAcousticProfile('new_user')
similar_profile = manager.get_profile('similar_user')

# Initialize with 50% of similar user's baseline
new_profile.baseline_pitch = similar_profile.baseline_pitch * 0.5
new_profile.baseline_energy = similar_profile.baseline_energy * 0.5

# Will quickly adapt to actual user
```

---

## File Structure

```
models/
├── personalized_acoustic_profiling.py   # Profiling system
└── audio_acoustic_encoder.py            # Base acoustic encoder

acoustic_profiles/                        # Stored profiles
├── alice_001.pkl
├── bob_002.pkl
└── charlie_003.pkl
```

---

## Future Enhancements

- [ ] Automatic speaker identification
- [ ] Emotion-specific adaptation rates
- [ ] Circadian rhythm modeling (time-of-day effects)
- [ ] Context-aware baselines (work vs. home voice)
- [ ] Federated learning across users
- [ ] Voice change detection (illness, stress)

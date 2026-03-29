#!/usr/bin/env python3
"""
MediaPipe Compatibility Test and Fix
"""

import sys

print("="*60)
print("MediaPipe Compatibility Test")
print("="*60)
print()

# Test MediaPipe import
print("1. Testing MediaPipe import...")
try:
    import mediapipe
    print(f"   ✓ MediaPipe version: {mediapipe.__version__}")
except ImportError as e:
    print(f"   ✗ MediaPipe import failed: {e}")
    sys.exit(1)

# Test solutions attribute
print("\n2. Testing mediapipe.solutions...")
try:
    from mediapipe import solutions
    print(f"   ✓ mediapipe.solutions available")
    print(f"   Available solutions: {dir(solutions)[:10]}...")
except AttributeError as e:
    print(f"   ✗ mediapipe.solutions not found: {e}")
    print("   Trying alternative import...")
    
# Test direct import
print("\n3. Testing direct FaceMesh import...")
try:
    from mediapipe.python.solutions import face_mesh
    print(f"   ✓ Direct import works: {face_mesh}")
except ImportError as e:
    print(f"   ✗ Direct import failed: {e}")

# Test legacy import
print("\n4. Testing legacy import...")
try:
    import mediapipe.python.solutions.face_mesh as mp_face_mesh
    import mediapipe.python.solutions.drawing_utils as mp_drawing
    print(f"   ✓ Legacy import works")
except ImportError as e:
    print(f"   ✗ Legacy import failed: {e}")

# Test creating FaceMesh
print("\n5. Testing FaceMesh instantiation...")
try:
    # Try new API first
    try:
        mp_face_mesh_module = mediapipe.solutions.face_mesh
        face_mesh = mp_face_mesh_module.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✓ FaceMesh created successfully (new API)")
        face_mesh.close()
    except:
        # Try direct import
        from mediapipe.python.solutions.face_mesh import FaceMesh
        face_mesh = FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print(f"   ✓ FaceMesh created successfully (direct import)")
        face_mesh.close()
except Exception as e:
    print(f"   ✗ FaceMesh creation failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*60)
print("Test Complete")
print("="*60)

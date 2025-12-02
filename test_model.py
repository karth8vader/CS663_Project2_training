#!/usr/bin/env python3
"""
Test script for the squat quality scoring model.

Usage:
    # Test on video file
    python test_model.py <video_path>
    python test_model.py <video_path> --model model.keras
    
    # Test on live camera stream
    python test_model.py --live
    python test_model.py --live --camera 0 --model model.keras
"""

import argparse
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

if TYPE_CHECKING:
    from tensorflow import keras

# Constants from training notebook (must match train_model.py)
NUM_FRAMES = 24  # Updated to match training script
NUM_LANDMARKS = 33  # BlazePose outputs 33 landmarks
LANDMARK_DIMS = 4   # x, y, z, visibility
SCORE_SCALE = 100.0  # labels are 0-100; model trains on 0-1 internally
FEATURE_DIMS = NUM_LANDMARKS * LANDMARK_DIMS + 1

# Exercise types (must match train_model.py)
EXERCISE_TYPES = [
    "squat", "pushup", "pullup", "lunge", "plank", "burpee",
    "jumping_jack", "mountain_climber", "situp", "deadlift",
    "bench_press", "dumbbell_curl", "overhead_press", "other"
]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def _sample_frame_indices(total_frames: int, num_target: int) -> np.ndarray:
    """Uniformly sample frame indices."""
    if total_frames <= 0:
        return np.zeros((num_target,), dtype=np.int32)
    idxs = np.linspace(0, max(total_frames - 1, 0), num_target).astype(np.int32)
    return idxs


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Robust normalization.

    Rules:
    - Use visibility (4th channel) to decide which landmarks to trust.
    - Prefer hip midpoint + shoulder midpoint for center/scale when available.
    - Fallback to visible-point mean if those keypoints are missing.
    - Rotate using hip vector if available, otherwise shoulders, otherwise PCA of visible points.

    Returns normalized landmarks (same shape) or None if insufficient visible points.
    """
    # landmarks shape: (NUM_LANDMARKS, LANDMARK_DIMS)
    vis = landmarks[:, 3] >= 0.25
    if vis.sum() < 2:
        # Not enough points
        return None

    def valid_pair(i, j):
        try:
            return bool(vis[i] and vis[j])
        except Exception:
            return False

    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12

    if valid_pair(LEFT_HIP, RIGHT_HIP):
        left_hip, right_hip = landmarks[LEFT_HIP, :3], landmarks[RIGHT_HIP, :3]
        center_hip = (left_hip + right_hip) / 2.0
    else:
        visible_coords = landmarks[vis, :3]
        center_hip = visible_coords.mean(axis=0)

    if valid_pair(LEFT_SHOULDER, RIGHT_SHOULDER):
        left_sh, right_sh = landmarks[LEFT_SHOULDER, :3], landmarks[RIGHT_SHOULDER, :3]
        center_sh = (left_sh + right_sh) / 2.0
    elif valid_pair(LEFT_HIP, RIGHT_HIP):
        # crude fallback: estimate shoulders above hips
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])
    else:
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])

    torso = np.linalg.norm(center_sh[:2] - center_hip[:2])
    hip_dist = np.linalg.norm(landmarks[LEFT_HIP, :2] - landmarks[RIGHT_HIP, :2]) if valid_pair(LEFT_HIP, RIGHT_HIP) else torso
    scale = max(torso, hip_dist, 1e-3)

    landmarks[:, :3] = (landmarks[:, :3] - center_hip) / scale

    # compute rotation
    if valid_pair(LEFT_HIP, RIGHT_HIP):
        hip_vec = (landmarks[RIGHT_HIP, :2] - landmarks[LEFT_HIP, :2])
    elif valid_pair(LEFT_SHOULDER, RIGHT_SHOULDER):
        hip_vec = (landmarks[RIGHT_SHOULDER, :2] - landmarks[LEFT_SHOULDER, :2])
    else:
        visible_pts = landmarks[vis, :2]
        if visible_pts.shape[0] < 2:
            hip_vec = np.array([1.0, 0.0])
        else:
            pts_centered = visible_pts - visible_pts.mean(axis=0)
            u, s, vh = np.linalg.svd(pts_centered, full_matrices=False)
            hip_vec = vh[0]

    angle = np.arctan2(hip_vec[1], hip_vec[0] + 1e-6)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    landmarks[:, :2] = landmarks[:, :2] @ rot.T
    return landmarks


def extract_keypoints(video_path: str, cache_dir: Path = None) -> np.ndarray:
    """Extract normalized keypoints for NUM_FRAMES timesteps.

    - Uses a local MediaPipe Pose instance (safe for parallel runs)
    - Marks missing frames and interpolates them
    - Appends a per-frame validity scalar (1.0 if frame originally had pose) as extra feature
    - Optionally caches outputs to speed repeated runs
    """
    if cache_dir is None:
        cache_dir = Path("checkpoints") / "keypoints_cache"
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(video_path.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{h}.npy"
    
    if cache_file.exists():
        try:
            return np.load(str(cache_file))
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    num_frames = len(frames)
    keypoints = np.full((NUM_FRAMES, NUM_LANDMARKS, LANDMARK_DIMS), np.nan, dtype=np.float32)
    
    if num_frames == 0:
        # nothing to do — return zeros + invalid mask
        filled = np.zeros((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS), dtype=np.float32)
        valid_mask = np.zeros((NUM_FRAMES,), dtype=np.float32)
        out = np.concatenate([filled, valid_mask[:, None]], axis=1)
        try:
            np.save(str(cache_file), out)
        except Exception:
            pass
        return out

    idxs = _sample_frame_indices(num_frames, NUM_FRAMES)

    mp_pose_local = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    valid_per_frame = np.zeros((NUM_FRAMES,), dtype=bool)
    for out_i, frame_idx in enumerate(idxs):
        frame = frames[int(frame_idx)]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_local.process(image_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            norm = _normalize_landmarks(coords)
            if norm is not None:
                keypoints[out_i] = norm
                valid_per_frame[out_i] = True

    try:
        mp_pose_local.close()
    except Exception:
        pass

    # interpolate missing values per-landmark and per-dimension
    idxs_time = np.arange(NUM_FRAMES)
    for li in range(NUM_LANDMARKS):
        for d in range(LANDMARK_DIMS):
            series = keypoints[:, li, d]
            good = ~np.isnan(series)
            if good.any():
                keypoints[:, li, d] = np.interp(idxs_time, idxs_time[good], series[good])
            else:
                keypoints[:, li, d] = 0.0

    frame_valid = valid_per_frame.astype(np.float32)

    keypoints_flat = keypoints.reshape((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS))
    out = np.concatenate([keypoints_flat, frame_valid[:, None]], axis=1).astype(np.float32)

    try:
        np.save(str(cache_file), out)
    except Exception:
        pass

    return out


def extract_keypoints_from_frames(frames: list, mp_pose_instance) -> np.ndarray:
    """Extract normalized keypoints from a list of frames.
    
    Args:
        frames: List of video frames (BGR format)
        mp_pose_instance: MediaPipe Pose instance
        
    Returns:
        Keypoints array of shape (NUM_FRAMES, FEATURE_DIMS)
    """
    num_frames = len(frames)
    if num_frames == 0:
        filled = np.zeros((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS), dtype=np.float32)
        valid_mask = np.zeros((NUM_FRAMES,), dtype=np.float32)
        return np.concatenate([filled, valid_mask[:, None]], axis=1).astype(np.float32)
    
    # Sample frames uniformly
    idxs = _sample_frame_indices(num_frames, NUM_FRAMES)
    keypoints = np.full((NUM_FRAMES, NUM_LANDMARKS, LANDMARK_DIMS), np.nan, dtype=np.float32)
    valid_per_frame = np.zeros((NUM_FRAMES,), dtype=bool)
    
    for out_i, frame_idx in enumerate(idxs):
        frame = frames[int(frame_idx)]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose_instance.process(image_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            norm = _normalize_landmarks(coords)
            if norm is not None:
                keypoints[out_i] = norm
                valid_per_frame[out_i] = True
    
    # Interpolate missing values
    idxs_time = np.arange(NUM_FRAMES)
    for li in range(NUM_LANDMARKS):
        for d in range(LANDMARK_DIMS):
            series = keypoints[:, li, d]
            good = ~np.isnan(series)
            if good.any():
                keypoints[:, li, d] = np.interp(idxs_time, idxs_time[good], series[good])
            else:
                keypoints[:, li, d] = 0.0
    
    frame_valid = valid_per_frame.astype(np.float32)
    keypoints_flat = keypoints.reshape((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS))
    out = np.concatenate([keypoints_flat, frame_valid[:, None]], axis=1).astype(np.float32)
    return out


def predict_score_from_keypoints(model, keypoints: np.ndarray, is_tflite: bool = False, interpreter=None) -> tuple:
    """Predict quality score and exercise type from keypoints.
    
    Args:
        model: Loaded Keras model (if not TFLite) or None (if TFLite)
        keypoints: Keypoints array of shape (NUM_FRAMES, FEATURE_DIMS)
        is_tflite: Whether using TFLite model
        interpreter: TFLite interpreter (if using TFLite)
        
    Returns:
        Tuple of (score: float, exercise_type: str, confidence: float)
        If model doesn't support exercise type, exercise_type will be None
    """
    keypoints = keypoints.reshape(1, NUM_FRAMES, FEATURE_DIMS)
    
    if is_tflite and interpreter is not None:
        # TFLite inference
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], keypoints.astype(np.float32))
        interpreter.invoke()
        
        # Get outputs
        if len(output_details) == 2:
            # Multi-output model (exercise type + score)
            exercise_probs = interpreter.get_tensor(output_details[0]['index'])
            score_norm = interpreter.get_tensor(output_details[1]['index'])[0][0]
            
            # Get predicted exercise type
            exercise_idx = np.argmax(exercise_probs[0])
            exercise_type = EXERCISE_TYPES[exercise_idx]
            confidence = float(exercise_probs[0][exercise_idx])
            
            return score_norm * SCORE_SCALE, exercise_type, confidence
        else:
            # Single output (score only)
            score_norm = interpreter.get_tensor(output_details[0]['index'])[0][0]
            return score_norm * SCORE_SCALE, None, 0.0
    else:
        # Keras model inference
        predictions = model.predict(keypoints, verbose=0)
        
        if isinstance(predictions, dict):
            # Multi-output model
            score_norm = float(predictions['score_output'][0][0])
            exercise_probs = predictions['exercise_type_output'][0]
            exercise_idx = np.argmax(exercise_probs)
            exercise_type = EXERCISE_TYPES[exercise_idx]
            confidence = float(exercise_probs[exercise_idx])
            return score_norm * SCORE_SCALE, exercise_type, confidence
        else:
            # Single output
            score_norm = float(predictions[0][0])
            return score_norm * SCORE_SCALE, None, 0.0


def predict_score(model, video_path: str, is_tflite: bool = False, interpreter=None) -> tuple:
    """Predict quality score and exercise type for a video.
    
    Args:
        model: Loaded Keras model (if not TFLite) or None (if TFLite)
        video_path: Path to video file
        is_tflite: Whether using TFLite model
        interpreter: TFLite interpreter (if using TFLite)
        
    Returns:
        Tuple of (score: float, exercise_type: str, confidence: float)
    """
    keypoints = extract_keypoints(video_path)
    return predict_score_from_keypoints(model, keypoints, is_tflite, interpreter)


def run_live_stream(model, camera_id: int = 0, is_tflite: bool = False, interpreter=None):
    """Run live video stream with real-time exercise scoring and detection.
    
    Args:
        model: Loaded Keras model (if not TFLite) or None (if TFLite)
        camera_id: Camera device ID (default: 0)
        is_tflite: Whether using TFLite model
        interpreter: TFLite interpreter (if using TFLite)
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera {camera_id}")
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    mp_pose_instance = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
    )
    
    # Frame buffer for accumulating NUM_FRAMES frames
    frame_buffer = []
    current_score = None
    current_exercise = None
    current_confidence = None
    fps_counter = 0
    fps_start_time = cv2.getTickCount()
    
    print("\n" + "="*50)
    print("Live Exercise Quality Scoring")
    print("="*50)
    print("Press 'q' to quit")
    print("Make sure you're visible in the camera frame!")
    print("="*50 + "\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame with MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = mp_pose_instance.process(image_rgb)
            
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Add frame to buffer
            frame_buffer.append(frame.copy())
            
            # Keep only last NUM_FRAMES frames
            if len(frame_buffer) > NUM_FRAMES:
                frame_buffer.pop(0)
            
            # Run inference when we have enough frames
            if len(frame_buffer) == NUM_FRAMES:
                try:
                    keypoints = extract_keypoints_from_frames(frame_buffer, mp_pose_instance)
                    current_score, current_exercise, current_confidence = predict_score_from_keypoints(
                        model, keypoints, is_tflite, interpreter
                    )
                except Exception as e:
                    print(f"Inference error: {e}")
            
            # Display score and info
            h, w = frame.shape[:2]
            
            # Score and exercise display
            if current_score is not None:
                score_text = f"Score: {current_score:.1f} / {int(SCORE_SCALE)}"
                color = (0, 255, 0) if current_score >= 70 else (0, 165, 255) if current_score >= 50 else (0, 0, 255)
                
                # Build text lines
                lines = [score_text]
                if current_exercise:
                    exercise_display = current_exercise.replace("_", " ").title()
                    if current_confidence:
                        lines.append(f"Exercise: {exercise_display} ({current_confidence*100:.1f}%)")
                    else:
                        lines.append(f"Exercise: {exercise_display}")
                
                # Calculate text dimensions
                line_height = 35
                max_width = 0
                for line in lines:
                    (text_width, text_height), _ = cv2.getTextSize(
                        line, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )
                    max_width = max(max_width, text_width)
                
                # Background rectangle
                total_height = len(lines) * line_height + 20
                cv2.rectangle(
                    frame,
                    (10, 10),
                    (max_width + 40, total_height),
                    (0, 0, 0),
                    -1
                )
                
                # Draw text lines
                y_offset = 35
                for i, line in enumerate(lines):
                    if i == 0:
                        # Score line with color
                cv2.putText(
                    frame,
                            line,
                            (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                            1.0,
                    color,
                            2,
                            cv2.LINE_AA
                        )
                    else:
                        # Exercise line in white
                        cv2.putText(
                            frame,
                            line,
                            (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                    cv2.LINE_AA
                )
                    y_offset += line_height
            else:
                buffer_text = f"Buffering frames: {len(frame_buffer)}/{NUM_FRAMES}"
                cv2.putText(
                    frame,
                    buffer_text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # FPS counter
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = cv2.getTickCount()
                fps = 30.0 / ((fps_end_time - fps_start_time) / cv2.getTickFrequency())
                fps_start_time = fps_end_time
                fps_counter = 0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
            
            # Instructions
            cv2.putText(
                frame,
                "Press 'q' to quit",
                (w - 200, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            cv2.imshow('Exercise Quality Scorer - Live', frame)
            
            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            mp_pose_instance.close()
        except Exception:
            pass
        print("\nStream ended")


def main():
    parser = argparse.ArgumentParser(description="Test squat quality scoring model")
    parser.add_argument("video_path", type=str, nargs="?", default=None,
                       help="Path to video file to test (optional if using --live)")
    parser.add_argument("--model", type=str, default="checkpoints/squat_scorer.keras", 
                       help="Path to model file (.keras or .tflite)")
    parser.add_argument("--no-cache", action="store_true",
                       help="Disable keypoint caching")
    parser.add_argument("--live", action="store_true",
                       help="Use live camera stream instead of video file")
    parser.add_argument("--camera", type=int, default=0,
                       help="Camera device ID for live stream (default: 0)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.live and args.video_path is None:
        parser.error("Either provide a video_path or use --live flag")
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return 1
    
    # Check if TFLite or Keras model
    is_tflite = model_path.suffix == '.tflite'
    model = None
    interpreter = None
    
    print(f"Loading model from: {model_path}")
    try:
        if is_tflite:
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print("TFLite model loaded successfully!")
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Outputs: {len(output_details)}")
            if len(output_details) == 2:
                print("  ✓ Multi-exercise model detected (score + exercise type)")
            else:
                print("  ✓ Single-output model (score only)")
        else:
            # Load Keras model
        try:
            import keras
        except ImportError:
            print("\n" + "="*60)
            print("ERROR: Keras dependencies not installed!")
            print("="*60)
            print("Please install required packages:")
            print("  pip install -r requirements.txt")
            print("\nOr install manually:")
            print("  pip install \"tensorflow<2.17\" mediapipe opencv-python numpy rich")
            print("="*60 + "\n")
            return 1
        
        model = tf.keras.models.load_model(str(model_path))
            print("Keras model loaded successfully!")
            # Check if multi-output
            if isinstance(model.output_shape, dict) or (isinstance(model.output_shape, list) and len(model.output_shape) > 1):
                print("  ✓ Multi-exercise model detected (score + exercise type)")
            else:
                print("  ✓ Single-output model (score only)")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        if not is_tflite:
        print("2. Check that TensorFlow version is < 2.17")
        print("3. Try: pip install --upgrade tensorflow keras")
        import traceback
        traceback.print_exc()
        return 1
    
    # Live stream mode
    if args.live:
        try:
            run_live_stream(model, camera_id=args.camera, is_tflite=is_tflite, interpreter=interpreter)
            return 0
        except Exception as e:
            print(f"Error during live stream: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    # Video file mode
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return 1
    
    print(f"\nProcessing video: {video_path}")
    print("Extracting pose keypoints...")
    
    try:
        cache_dir = None if args.no_cache else Path("checkpoints") / "keypoints_cache"
        keypoints = extract_keypoints(str(video_path), cache_dir=cache_dir)
        print(f"Keypoints extracted. Shape: {keypoints.shape}")
        
        print("\nRunning inference...")
        score, exercise_type, confidence = predict_score(model, str(video_path), is_tflite=is_tflite, interpreter=interpreter)
        
        print(f"\n{'='*50}")
        print(f"Predicted Quality Score: {score:.2f} / {int(SCORE_SCALE)}")
        if exercise_type:
            print(f"Detected Exercise: {exercise_type.replace('_', ' ').title()}")
            if confidence > 0:
                print(f"Confidence: {confidence*100:.1f}%")
        print(f"{'='*50}")
        
        return 0
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())


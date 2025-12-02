#!/usr/bin/env python3
"""
Score a video using a trained exercise-quality model.

This CLI supports scoring multiple exercise types: squat, pushup, and lunge.
It mirrors the preprocessing in the training notebooks so predictions match
the models trained in the repo.

Environment (tested):
  - Python 3.10+ (3.11 recommended)
  - pip install "tensorflow<2.17" mediapipe opencv-python numpy

Usage:
  python score_video.py /path/to/video.mp4

What this does:
  - Samples NUM_FRAMES frames uniformly from the video
  - Runs MediaPipe BlazePose for each sampled frame
  - Applies the same robust normalization as the notebook (visibility-aware, hip/shoulder centering, rotate)
  - Interpolates missing frames, appends a per-frame validity flag
    - Loads a model from checkpoints/<exercise>_scorer_savedmodel (preferred),
        or <exercise>_scorer.keras, or falls back to checkpoints/model.keras
  - Predicts a normalized value and prints a score in 0..100

Notes:
  - TFLite supported models are not used here. If you prefer tflite inference, adapt the code or call the Android helper.
  - This script mirrors training.ipynb's preprocessing so predictions match the notebook-trained model.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf

# Constants (must match notebooks and model input expectations)
SEED = 42
NUM_FRAMES = 16
NUM_LANDMARKS = 33
LANDMARK_DIMS = 4
SCORE_SCALE = 100.0

MODEL_DIR = Path("checkpoints")


def sample_frame_indices(total_frames: int, num_target: int) -> np.ndarray:
    """Return `num_target` frame indices evenly sampled from the range [0, total_frames-1].

    If the video is empty (total_frames == 0) we return all zeros. Downstream code
    expects indices (ints) and handles empty cases separately.
    """
    if total_frames <= 0:
        # preserve shape and dtype for callers
        return np.zeros((num_target,), dtype=np.int32)
    return np.linspace(0, max(total_frames - 1, 0), num_target).astype(np.int32)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Robust per-frame normalization for pose landmarks.

        - Input: landmarks array shape (NUM_LANDMARKS, LANDMARK_DIMS) where columns are
            (x, y, z, visibility).
        - Output: normalized landmarks in the same shape, or None if the frame has
            insufficient visible points (visibility threshold used).

        Normalization strategy (for reproducible model input):
        1) Use hip midpoint as center when both hips are visible, otherwise use mean of visible points.
        2) Estimate scale from torso length or hip distance.
        3) Rotate so the main body axis (hip vector or shoulder vector or PCA) is aligned horizontally.
    """
    vis = landmarks[:, 3] >= 0.25
    if vis.sum() < 2:
        return None

    def valid_pair(i, j):
        return bool(vis[i] and vis[j])

    LHIP, RHIP = 23, 24
    LSH, RSH = 11, 12

    if valid_pair(LHIP, RHIP):
        left_hip, right_hip = landmarks[LHIP, :3], landmarks[RHIP, :3]
        center_hip = (left_hip + right_hip) / 2.0
    else:
        visible_coords = landmarks[vis, :3]
        center_hip = visible_coords.mean(axis=0)

    if valid_pair(LSH, RSH):
        left_sh, right_sh = landmarks[LSH, :3], landmarks[RSH, :3]
        center_sh = (left_sh + right_sh) / 2.0
    elif valid_pair(LHIP, RHIP):
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])
    else:
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])

    torso = np.linalg.norm(center_sh[:2] - center_hip[:2])
    hip_dist = np.linalg.norm(landmarks[LHIP, :2] - landmarks[RHIP, :2]) if valid_pair(LHIP, RHIP) else torso
    scale = max(torso, hip_dist, 1e-3)

    # Copy landmarks to avoid mutating the input array
    out = landmarks.copy()
    out[:, :3] = (out[:, :3] - center_hip) / scale

    if valid_pair(LHIP, RHIP):
        vec = out[RHIP, :2] - out[LHIP, :2]
    elif valid_pair(LSH, RSH):
        vec = out[RSH, :2] - out[LSH, :2]
    else:
        pts = out[vis, :2]
        if pts.shape[0] < 2:
            vec = np.array([1.0, 0.0])
        else:
            pts_centered = pts - pts.mean(axis=0)
            u, s, vh = np.linalg.svd(pts_centered, full_matrices=False)
            vec = vh[0]

    # small epsilon avoids exact division by zero when computing angle
    angle = np.arctan2(vec[1], vec[0] + 1e-6)
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    out[:, :2] = out[:, :2] @ rot.T
    return out


def extract_keypoints_np(video_path: str) -> np.ndarray:
    """Extract normalized per-frame keypoint features for NUM_FRAMES timesteps.

    Output is a (NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS + 1) array where the
    last column is a per-frame validity flag (1.0 = original frame had pose, 0.0 = interpolated).
    The function caches results to `checkpoints/keypoints_cache/` using a sha1 hash of path
    so repeated runs are fast.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    num_frames = len(frames)
    # keypoints array holds (x,y,z,visibility) per landmark per output timestep;
    # filled with NaN so we can detect missing values and interpolate later
    keypoints = np.full((NUM_FRAMES, NUM_LANDMARKS, LANDMARK_DIMS), np.nan, dtype=np.float32)
    if num_frames == 0:
        # empty output shape: flatten + valid flag
        flat = np.zeros((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS + 1), dtype=np.float32)
        return flat

    idxs = sample_frame_indices(num_frames, NUM_FRAMES)
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    # valid_per_frame marks frames where MediaPipe returned a valid normalized pose
    valid_per_frame = np.zeros((NUM_FRAMES,), dtype=bool)
    for out_i, frame_idx in enumerate(idxs):
        frame = frames[int(frame_idx)]
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_pose.process(image_rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            coords = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            norm = normalize_landmarks(coords)
            if norm is not None:
                keypoints[out_i] = norm
                valid_per_frame[out_i] = True

    try:
        mp_pose.close()
    except Exception:
        pass

    # interpolate missing values for each landmark/dimension across the temporal axis
    # if a series has no valid points we fill zeros
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

    # Flatten and append the per-frame validity flag as an extra feature column
    keypoints_flat = keypoints.reshape((NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS))
    out = np.concatenate([keypoints_flat, frame_valid[:, None]], axis=1).astype(np.float32)
    return out


def load_model_for_inference(model_dir: Path, exercise: str = "squat"):
    """Load a model for the given exercise.

    Lookup order (best->fallback):
      1) SavedModel dir: <exercise>_scorer_savedmodel/
      2) Exported single-file Keras: <exercise>_scorer.keras
      3) Checkpoint file: model.keras (existing project checkpoint)
    """
    # prefer savedmodel dir first (more robust for conversion)
    saved = model_dir / f"{exercise}_scorer_savedmodel"
    exported = model_dir / f"{exercise}_scorer.keras"
    checkpoint = model_dir / "model.keras"

    if saved.exists() and saved.is_dir():
        print(f"Loading SavedModel from {saved}")
        return tf.keras.models.load_model(str(saved))

    # If SavedModel not present, prefer the `.keras` archive (native Keras format)
    # produced by the notebooks.
    if exported.exists():
        print(f"Loading Keras model from {exported}")
        return tf.keras.models.load_model(str(exported))
    # Finally fallback to the generic checkpoint file used during training
    if checkpoint.exists():
        print(f"Loading model from checkpoint {checkpoint}")
        return tf.keras.models.load_model(str(checkpoint))
    raise FileNotFoundError(
        "No saved model found under checkpoints/. Expected one of: <exercise>_scorer_savedmodel/, <exercise>_scorer.keras, or model.keras"
    )


def score_video(video_path: str, exercise: str = "squat") -> float:
    kpts = extract_keypoints_np(video_path)
    # input shape expected: (1, NUM_FRAMES, NUM_LANDMARKS*LANDMARK_DIMS+1)
    inp = kpts.reshape(1, NUM_FRAMES, NUM_LANDMARKS * LANDMARK_DIMS + 1)

    model = load_model_for_inference(MODEL_DIR, exercise=exercise)
    pred = float(model.predict(inp, verbose=0)[0][0])
    return float(max(0.0, min(SCORE_SCALE, pred * SCORE_SCALE)))


def main():
    parser = argparse.ArgumentParser(description="Score a squat video using the trained model")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--exercise", "-e", choices=["squat", "pushup", "lunge"], default="squat",
                        help="Which model / exercise to use (squat|pushup|lunge). Default: squat")
    args = parser.parse_args()

    score = score_video(args.video, exercise=args.exercise)
    print(f"Score: {score:.2f} / {int(SCORE_SCALE)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/score_video.py /path/to/video.mp4")
        sys.exit(1)
    main()

#!/usr/bin/env python3
"""
Improved Squat Quality Scoring Model Training Script

This script trains a BiLSTM-based model to predict squat quality scores (0-100)
from pose keypoints extracted from videos using MediaPipe.

Usage:
    python train_model.py
    python train_model.py --epochs 100 --batch-size 8
"""

import argparse
import csv
import hashlib
import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Try to import fiftyone for UCF-101 dataset
try:
    import fiftyone.zoo as foz
    try:
        from fiftyone import ViewField as F
    except ImportError:
        # Older versions might not have ViewField
        F = None
    HAS_FIFTYONE = True
except ImportError:
    HAS_FIFTYONE = False
    F = None

# Try to import matplotlib for plotting (optional)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Matplotlib not available - plots will be skipped")

# ============================================================================
# Configuration
# ============================================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# IMPROVED: Better hyperparameters
BATCH_SIZE = 8  # Increased from 4 for more stable gradients
NUM_FRAMES = 24  # Increased from 16 for more temporal context
IMG_SIZE = 160
NUM_LANDMARKS = 33  # BlazePose outputs 33 landmarks
LANDMARK_DIMS = 4   # x, y, z, visibility
SCORE_SCALE = 100.0  # labels are 0-100; model trains on 0-1 internally
FEATURE_DIMS = NUM_LANDMARKS * LANDMARK_DIMS + 1

DATA_ROOT = Path("data")
# Support both old and new directory names for backward compatibility
TRAIN_DIR = DATA_ROOT / "exercises_train"  # Updated to support multiple exercises
TEST_DIR = DATA_ROOT / "exercises_test"    # Updated to support multiple exercises
OLD_TRAIN_DIR = DATA_ROOT / "squats_train"  # Legacy support
OLD_TEST_DIR = DATA_ROOT / "squats_test"    # Legacy support
LABELS_PATH = DATA_ROOT / "exercise_scores.csv"  # Updated filename
OLD_LABELS_PATH = DATA_ROOT / "squat_scores.csv"  # Legacy support
MODEL_DIR = Path("checkpoints")

# Supported exercise types
EXERCISE_TYPES = [
    "squat", "pushup", "pullup", "lunge", "plank", "burpee",
    "jumping_jack", "mountain_climber", "situp", "deadlift",
    "bench_press", "dumbbell_curl", "overhead_press", "other"
]

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")

mp_pose = mp.solutions.pose


# ============================================================================
# Dataset Download Functions
# ============================================================================

def download_ucf101_all_workouts(output_dir: Path = None, split_train_test: bool = True):
    """
    Download UCF-101 dataset using FiftyOne and extract ALL workout/exercise classes.
    
    This includes: BodyWeightSquats, PushUps, PullUps, JumpRope, JumpingJack, etc.
    
    Requirements:
    - Python 3.11 (fiftyone requirement)
    - pip install fiftyone
    
    Args:
        output_dir: Directory to save the dataset
        split_train_test: Whether to split into train/test (80/20)
    
    Returns:
        True if successful, False otherwise
    """
    if not HAS_FIFTYONE:
        print("\n" + "="*60)
        print("ERROR: FiftyOne library not installed!")
        print("="*60)
        print("To download UCF-101 dataset:")
        print("  1. Ensure you have Python 3.11")
        print("  2. Run: pip install fiftyone")
        print("  3. Run this script again with --download-ucf101")
        print("="*60 + "\n")
        return False
    
    # Check Python version
    if sys.version_info < (3, 11):
        print("\n" + "="*60)
        print("WARNING: Python 3.11+ recommended for FiftyOne")
        print(f"Current version: {sys.version}")
        print("="*60)
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    if output_dir is None:
        output_dir = DATA_ROOT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "squats_train"
    test_dir = output_dir / "squats_test"
    
    print("\n" + "="*60)
    print("DOWNLOADING UCF-101 DATASET - ALL WORKOUTS")
    print("="*60)
    print("This will download ~50GB of data")
    print("Extracting ALL workout/exercise classes for maximum dataset size")
    print("The download may take a while depending on your internet connection...")
    print("="*60 + "\n")
    
    try:
        # Load UCF-101 dataset using FiftyOne
        print("Loading UCF-101 dataset from FiftyOne Zoo...")
        print("(This may take a while - downloading ~50GB)")
        
        dataset = foz.load_zoo_dataset("ucf101")
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Get all available workout/exercise classes
        print("\nFinding all workout/exercise classes...")
        
        # Common workout/exercise classes in UCF-101
        workout_classes = [
            "BodyWeightSquats",
            "PushUps",
            "PullUps", 
            "JumpRope",
            "JumpingJack",
            "Lunges",
            "Burpees",
            "Plank",
            "MountainClimber",
            "SitUps",
            "WallPushups",
            "HandstandPushups",
            "DumbbellCurls",
            "DumbbellPress",
            "BenchPress",
            "Deadlift",
            "BarbellSquats",
        ]
        
        # Try to get all available labels first
        all_labels = []
        try:
            if hasattr(dataset, 'distinct'):
                all_labels = dataset.distinct("ground_truth.label")
                print(f"Found {len(all_labels)} total classes in dataset")
        except:
            pass
        
        # Filter for workout classes (case-insensitive matching)
        workout_dataset = None
        found_classes = []
        
        for workout_class in workout_classes:
            # Try different filtering methods
            try:
                # Method 1: Exact match on label
                filtered = dataset.match({"ground_truth.label": workout_class})
                if len(filtered) > 0:
                    if workout_dataset is None:
                        workout_dataset = filtered
                    else:
                        workout_dataset = workout_dataset | filtered
                    found_classes.append(workout_class)
                    print(f"  ✓ {workout_class}: {len(filtered)} videos")
            except:
                pass
            
            # Method 2: Case-insensitive match
            if workout_class.lower() not in [c.lower() for c in found_classes]:
                try:
                    for label in all_labels:
                        if workout_class.lower() in label.lower() or label.lower() in workout_class.lower():
                            filtered = dataset.match({"ground_truth.label": label})
                            if len(filtered) > 0:
                                if workout_dataset is None:
                                    workout_dataset = filtered
                                else:
                                    workout_dataset = workout_dataset | filtered
                                found_classes.append(label)
                                print(f"  ✓ {label}: {len(filtered)} videos")
                                break
                except:
                    pass
        
        # If no specific classes found, try to get all exercise-related classes
        if workout_dataset is None or len(workout_dataset) == 0:
            print("\n⚠ No specific workout classes found. Trying to find exercise-related classes...")
            exercise_keywords = ["squat", "push", "pull", "jump", "lunge", "burpee", "plank", 
                               "curl", "press", "lift", "situp", "exercise", "workout"]
            for keyword in exercise_keywords:
                try:
                    for label in all_labels:
                        if keyword.lower() in label.lower():
                            filtered = dataset.match({"ground_truth.label": label})
                            if len(filtered) > 0:
                                if workout_dataset is None:
                                    workout_dataset = filtered
                                else:
                                    workout_dataset = workout_dataset | filtered
                                if label not in found_classes:
                                    found_classes.append(label)
                                    print(f"  ✓ {label}: {len(filtered)} videos")
                except:
                    pass
        
        if workout_dataset is None or len(workout_dataset) == 0:
            print("\n⚠ No workout videos found!")
            print("Available classes:", all_labels[:20] if all_labels else "Could not determine")
            print("\nUsing ALL videos from dataset instead...")
            workout_dataset = dataset
        
        print(f"\n✓ Found {len(workout_dataset)} total workout videos")
        print(f"  Classes: {', '.join(found_classes[:10])}{'...' if len(found_classes) > 10 else ''}")
        
        # Create output directories
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Export videos
        print(f"\nExporting videos to {output_dir}...")
        
        # Get all video file paths
        video_paths = []
        for sample in workout_dataset:
            filepath = sample.filepath
            if filepath:
                filepath = Path(filepath)
                if filepath.exists():
                    video_paths.append(str(filepath))
                else:
                    # Try to get the actual file path from FiftyOne
                    try:
                        # Sometimes filepath might be relative or in FiftyOne's cache
                        actual_path = sample.metadata.get('filepath', filepath)
                        if Path(actual_path).exists():
                            video_paths.append(str(actual_path))
                    except:
                        pass
        
        if len(video_paths) == 0:
            print("\n⚠ No accessible video files found!")
            print("The videos might be in FiftyOne's cache. Trying alternative export method...")
            # Try exporting the dataset
            try:
                export_dir = output_dir / "ucf101_export"
                export_dir.mkdir(parents=True, exist_ok=True)
                workout_dataset.export(
                    export_dir=str(export_dir),
                    dataset_type=foz.types.VideoClassificationDirectoryTree,
                )
                # Find videos in export directory (search for all workout classes)
                for workout_class in found_classes[:5]:  # Search first 5 found classes
                    for video_file in export_dir.rglob(f"{workout_class}/**/*"):
                        if video_file.suffix.lower() in VIDEO_EXTS:
                            video_paths.append(str(video_file))
                print(f"✓ Exported {len(video_paths)} videos to {export_dir}")
            except Exception as e:
                print(f"Export failed: {e}")
                return False
        
        print(f"Found {len(video_paths)} accessible video files")
        
        # Split into train/test (80/20)
        if split_train_test:
            random.shuffle(video_paths)
            split_idx = int(0.8 * len(video_paths))
            train_videos = video_paths[:split_idx]
            test_videos = video_paths[split_idx:]
            
            print(f"\nCopying {len(train_videos)} videos to {train_dir}...")
            for i, video_path in enumerate(train_videos, 1):
                dest = train_dir / Path(video_path).name
                shutil.copy2(video_path, dest)
                if i % 10 == 0:
                    print(f"  Copied {i}/{len(train_videos)} videos...")
            
            print(f"\nCopying {len(test_videos)} videos to {test_dir}...")
            for i, video_path in enumerate(test_videos, 1):
                dest = test_dir / Path(video_path).name
                shutil.copy2(video_path, dest)
                if i % 10 == 0:
                    print(f"  Copied {i}/{len(test_videos)} videos...")
        else:
            # Put all in train
            print(f"\nCopying {len(video_paths)} videos to {train_dir}...")
            for i, video_path in enumerate(video_paths, 1):
                dest = train_dir / Path(video_path).name
                shutil.copy2(video_path, dest)
                if i % 10 == 0:
                    print(f"  Copied {i}/{len(video_paths)} videos...")
        
        print("\n" + "="*60)
        print("✓ UCF-101 Workout dataset downloaded successfully!")
        print("="*60)
        if split_train_test:
            print(f"Train videos: {len(train_videos)}")
            print(f"Test videos: {len(test_videos)}")
        else:
            print(f"Total videos: {len(video_paths)}")
        print(f"\nNext steps:")
        print("  1. Review the videos in data/exercises_train/ and data/exercises_test/")
        print("  2. Run the training script again to generate the label CSV")
        print("  3. Fill in scores (0-100) and exercise_type in data/exercise_scores.csv")
        print("  4. Run training: python train_model.py")
        print("="*60 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading UCF-101 dataset: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Ensure you have Python 3.11+")
        print("  2. Install fiftyone: pip install fiftyone")
        print("  3. Check your internet connection")
        print("  4. Ensure you have ~50GB free disk space")
        return False


def download_coco_pose_dataset(output_dir: Path = None, split_train_test: bool = True):
    """
    Download COCO Pose dataset for multi-exercise training.
    
    COCO Pose dataset includes diverse human activities suitable for exercise recognition.
    
    Args:
        output_dir: Directory to save the dataset
        split_train_test: Whether to split into train/test (80/20)
    
    Returns:
        True if successful, False otherwise
    """
    if output_dir is None:
        output_dir = DATA_ROOT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    coco_dir = output_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("COCO POSE DATASET SETUP")
    print("="*60)
    print("COCO Pose dataset provides diverse human pose annotations")
    print("suitable for training on multiple exercise types.")
    print("\nNote: COCO uses 17 keypoints (MediaPipe uses 33)")
    print("You may need to adapt the model or convert keypoints.")
    print("="*60 + "\n")
    
    try:
        import json
        import requests
        from tqdm import tqdm
        
        # COCO dataset URLs
        coco_base_url = "http://images.cocodataset.org"
        annotations_url = f"{coco_base_url}/annotations/person_keypoints_train2017.json"
        val_annotations_url = f"{coco_base_url}/annotations/person_keypoints_val2017.json"
        
        print("Downloading COCO annotations...")
        print("(Images must be downloaded manually due to size)")
        
        # Download annotations
        for url, filename in [
            (annotations_url, "person_keypoints_train2017.json"),
            (val_annotations_url, "person_keypoints_val2017.json")
        ]:
            filepath = coco_dir / filename
            if filepath.exists():
                print(f"  ✓ {filename} already exists, skipping...")
                continue
            
            print(f"  Downloading {filename}...")
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                with open(filepath, 'wb') as f, tqdm(
                    desc=filename,
                    total=total_size,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
                
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                return False
        
        print("\n" + "="*60)
        print("ANNOTATIONS DOWNLOADED")
        print("="*60)
        print("\nNext steps:")
        print("1. Download images manually from: https://cocodataset.org/#download")
        print("   Required files:")
        print("     - train2017.zip (~18GB)")
        print("     - val2017.zip (~1GB)")
        print("\n2. Extract images to:")
        print(f"     {coco_dir}/train2017/")
        print(f"     {coco_dir}/val2017/")
        print("\n3. Process annotations:")
        print("     python train_model.py --process-coco")
        print("\n" + "="*60)
        
        return True
        
    except ImportError:
        print("Error: Missing required packages")
        print("Install with: pip install requests tqdm")
        return False
    except Exception as e:
        print(f"\n✗ Error setting up COCO dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_coco_annotations(coco_dir: Path, output_dir: Path = None):
    """
    Process COCO pose annotations and convert to video format or extract keypoints.
    
    This function processes COCO JSON annotations and prepares them for training.
    Since COCO is image-based, we'll need to either:
    1. Convert sequences to videos, or
    2. Extract keypoints directly from images
    
    Args:
        coco_dir: Directory containing COCO dataset
        output_dir: Directory to save processed data
    """
    if output_dir is None:
        output_dir = DATA_ROOT
    
    print("\n" + "="*60)
    print("PROCESSING COCO POSE ANNOTATIONS")
    print("="*60)
    
    try:
        import json
        
        train_json = coco_dir / "person_keypoints_train2017.json"
        val_json = coco_dir / "person_keypoints_val2017.json"
        
        if not train_json.exists():
            print(f"Error: {train_json} not found")
            return False
        
        # Load annotations
        print("Loading COCO annotations...")
        with open(train_json, 'r') as f:
            train_data = json.load(f)
        
        if val_json.exists():
            with open(val_json, 'r') as f:
                val_data = json.load(f)
        else:
            val_data = None
        
        print(f"Loaded {len(train_data.get('annotations', []))} training annotations")
        if val_data:
            print(f"Loaded {len(val_data.get('annotations', []))} validation annotations")
        
        # Process annotations and extract keypoints
        # This would convert COCO format (17 keypoints) to MediaPipe format (33 keypoints)
        # or use COCO keypoints directly
        
        print("\nNote: COCO keypoints (17 points) differ from MediaPipe (33 points).")
        print("Consider using COCO keypoints directly or mapping them to MediaPipe format.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error processing COCO annotations: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_sample_dataset(output_dir: Path = None):
    """
    Download a sample dataset if no data exists.
    
    Note: This is a placeholder. Replace with actual dataset URL if available.
    For now, this will create the directory structure and provide instructions.
    """
    if output_dir is None:
        output_dir = DATA_ROOT
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "squats_train"
    test_dir = output_dir / "squats_test"
    
    # Check if data already exists
    if train_dir.exists() and (any(train_dir.rglob("*.mp4")) or any(train_dir.rglob("*.avi"))):
        print(f"Data already exists in {train_dir}")
        return False
    
    print("="*60)
    print("DATASET SETUP")
    print("="*60)
    print("\nNo dataset found. You have several options:")
    print("\n1. DOWNLOAD UCF-101 (Recommended):")
    print("   python train_model.py --download-ucf101")
    print("   - Downloads ~50GB, extracts only BodyWeightSquats")
    print("   - Requires Python 3.11+ and: pip install fiftyone")
    print("\n2. MANUAL SETUP:")
    print("   - Create folders: data/squats_train/ and data/squats_test/")
    print("   - Add your squat video files (.mp4, .mov, .avi, .mkv)")
    print("   - Run the script again to generate the label CSV")
    print("\n3. DOWNLOAD FROM URL:")
    print("   python train_model.py --download-dataset <URL>")
    print("\n" + "="*60)
    
    # Create empty directories
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreated directories:")
    print(f"  - {train_dir}")
    print(f"  - {test_dir}")
    print(f"\nPlease add your video files to these directories and run again.")
    
    return False


def download_from_url(url: str, output_path: Path, extract: bool = True):
    """
    Download a dataset from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        extract: Whether to extract if it's a zip file
    """
    print(f"Downloading from {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path, reporthook=_download_progress)
        print(f"\nDownloaded to {output_path}")
        
        if extract and output_path.suffix == '.zip':
            print(f"Extracting {output_path}...")
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_path.parent)
            print(f"Extracted to {output_path.parent}")
            # Optionally remove zip file
            # output_path.unlink()
        
        return True
    except Exception as e:
        print(f"Error downloading: {e}")
        return False


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    percent = min(downloaded * 100 / total_size, 100)
    bar_length = 40
    filled = int(bar_length * downloaded / total_size)
    bar = '=' * filled + '-' * (bar_length - filled)
    print(f'\r[{bar}] {percent:.1f}%', end='', flush=True)


# ============================================================================
# Data Loading Functions
# ============================================================================

def list_videos(root: Path):
    """List all video files in the directory."""
    return sorted(
        p for p in root.rglob("*")
        if p.suffix.lower() in VIDEO_EXTS and p.is_file()
    )


def ensure_label_file():
    """Create or update the label CSV file with exercise type support."""
    existing = {}
    # Check new format first
    if LABELS_PATH.exists():
        with LABELS_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("relative_path", "")
                existing[key] = {
                    "score": row.get("score", ""),
                    "exercise_type": row.get("exercise_type", "")
                }
    # Check old format for backward compatibility
    elif OLD_LABELS_PATH.exists():
        print(f"Found old label file: {OLD_LABELS_PATH}")
        print("Migrating to new format...")
        with OLD_LABELS_PATH.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row.get("relative_path", "")
                existing[key] = {
                    "score": row.get("score", ""),
                    "exercise_type": "squat"  # Default to squat for old data
                }

    rows = []
    # Check both new and old directories for backward compatibility
    video_dirs = []
    if TRAIN_DIR.exists():
        video_dirs.append(TRAIN_DIR)
    if TEST_DIR.exists():
        video_dirs.append(TEST_DIR)
    if OLD_TRAIN_DIR.exists():
        video_dirs.append(OLD_TRAIN_DIR)
        print(f"Found legacy directory: {OLD_TRAIN_DIR}")
    if OLD_TEST_DIR.exists():
        video_dirs.append(OLD_TEST_DIR)
        print(f"Found legacy directory: {OLD_TEST_DIR}")
    
    for video_dir in video_dirs:
        for p in list_videos(video_dir):
            rel = p.relative_to(DATA_ROOT).as_posix()
            # Try to infer exercise type from path/filename
            exercise_type = existing.get(rel, {}).get("exercise_type", "")
            if not exercise_type:
                # Infer from path
                path_lower = rel.lower()
                for ex_type in EXERCISE_TYPES:
                    if ex_type.replace("_", "") in path_lower or ex_type in path_lower:
                        exercise_type = ex_type
                        break
                if not exercise_type:
                    # Default based on directory
                    if "squat" in str(video_dir).lower():
                        exercise_type = "squat"
                    else:
                        exercise_type = "other"
            
            rows.append({
                "relative_path": rel,
                "exercise_type": exercise_type,
                "score": existing.get(rel, {}).get("score", "")
            })

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "exercise_type", "score"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Label file ready at {LABELS_PATH}.")
    print(f"Fill in 'score' (0-{int(SCORE_SCALE)}) and 'exercise_type' for each row.")
    print(f"Supported exercise types: {', '.join(EXERCISE_TYPES)}")
    return rows


def load_labeled_samples():
    """Load labeled video samples from CSV with exercise type support."""
    samples = []
    missing = []
    exercise_counts = {}
    
    with LABELS_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("relative_path", "")
            score_str = row.get("score", "").strip()
            exercise_type = row.get("exercise_type", "").strip().lower()
            
            if not rel:
                continue

            full = DATA_ROOT / rel
            if not full.exists():
                missing.append(rel)
                continue

            if not score_str:
                continue

            try:
                score = float(score_str)
            except ValueError:
                print(f"Skipping {rel}: invalid score '{score_str}'")
                continue

            score = max(0.0, min(SCORE_SCALE, score))
            
            # Validate exercise type
            if exercise_type and exercise_type not in EXERCISE_TYPES:
                # Try to map common variations
                exercise_type_map = {
                    "squats": "squat", "pushups": "pushup", "push-ups": "pushup",
                    "pullups": "pullup", "pull-ups": "pullup", "lunges": "lunge",
                    "burpees": "burpee", "jumping jacks": "jumping_jack",
                    "mountain climbers": "mountain_climber", "situps": "situp",
                    "sit-ups": "situp", "deadlifts": "deadlift", "bench press": "bench_press"
                }
                exercise_type = exercise_type_map.get(exercise_type, "other")
            
            if not exercise_type:
                exercise_type = "other"
            
            # Track exercise type distribution
            exercise_counts[exercise_type] = exercise_counts.get(exercise_type, 0) + 1
            
            # Store as (path, score, exercise_type)
            samples.append((str(full), score, exercise_type))

    if missing:
        print("Warning: paths not found on disk:", missing)

    print(f"Loaded {len(samples)} labeled samples.")
    if exercise_counts:
        print("Exercise type distribution:")
        for ex_type, count in sorted(exercise_counts.items(), key=lambda x: -x[1]):
            print(f"  {ex_type}: {count}")
    return samples


# ============================================================================
# Keypoint Extraction Functions
# ============================================================================

def _sample_frame_indices(total_frames: int, num_target: int) -> np.ndarray:
    """Uniformly sample frame indices."""
    if total_frames <= 0:
        return np.zeros((num_target,), dtype=np.int32)
    idxs = np.linspace(0, max(total_frames - 1, 0), num_target).astype(np.int32)
    return idxs


def _normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Robust normalization of pose landmarks."""
    vis = landmarks[:, 3] >= 0.25
    if vis.sum() < 2:
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
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])
    else:
        center_sh = center_hip + np.array([0.0, 0.5, 0.0])

    torso = np.linalg.norm(center_sh[:2] - center_hip[:2])
    hip_dist = np.linalg.norm(landmarks[LEFT_HIP, :2] - landmarks[RIGHT_HIP, :2]) if valid_pair(LEFT_HIP, RIGHT_HIP) else torso
    scale = max(torso, hip_dist, 1e-3)

    landmarks[:, :3] = (landmarks[:, :3] - center_hip) / scale

    # Compute rotation
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


def _extract_keypoints_np(video_path: str) -> np.ndarray:
    """Extract normalized keypoints for NUM_FRAMES timesteps."""
    cache_dir = MODEL_DIR / "keypoints_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(video_path.encode("utf-8")).hexdigest()
    cache_file = cache_dir / f"{h}.npy"
    
    if cache_file.exists():
        try:
            cached = np.load(str(cache_file))
            # If cached with different NUM_FRAMES, recompute
            if cached.shape[0] == NUM_FRAMES:
                return cached
        except Exception:
            pass

    cap = cv2.VideoCapture(video_path)
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

    try:
        np.save(str(cache_file), out)
    except Exception:
        pass

    return out


def augment_keypoints(keypoints: np.ndarray, training: bool = True) -> np.ndarray:
    """Apply data augmentation to keypoints during training."""
    if not training:
        return keypoints
    
    # Add small Gaussian noise to keypoints (not to validity flag)
    noise_std = 0.01
    keypoints_aug = keypoints.copy()
    keypoints_aug[:, :-1] += np.random.normal(0, noise_std, keypoints_aug[:, :-1].shape)
    return keypoints_aug.astype(np.float32)


# ============================================================================
# TensorFlow Dataset Pipeline
# ============================================================================

def load_keypoints(path: tf.Tensor) -> tf.Tensor:
    """Load keypoints from video path."""
    def _py_decode(p):
        return _extract_keypoints_np(p.numpy().decode("utf-8"))

    kpts = tf.py_function(_py_decode, [path], tf.float32)
    kpts.set_shape((NUM_FRAMES, FEATURE_DIMS))
    return kpts


def preprocess(path: tf.Tensor, score: tf.Tensor, exercise_type: tf.Tensor = None, training: bool = False, return_dict: bool = False) -> tuple:
    """Preprocess video path, score, and optionally exercise type."""
    keypoints = load_keypoints(path)
    
    # Apply augmentation during training
    if training:
        keypoints = tf.py_function(
            lambda k: augment_keypoints(k.numpy(), training=True),
            [keypoints],
            tf.float32
        )
        keypoints.set_shape((NUM_FRAMES, FEATURE_DIMS))
    
    score = tf.cast(score, tf.float32) / SCORE_SCALE
    score = tf.expand_dims(score, axis=-1)
    
    if exercise_type is not None:
        exercise_type = tf.cast(exercise_type, tf.int32)
        if return_dict:
            # Return dict format for multi-output models
            return keypoints, {"score_output": score, "exercise_type_output": exercise_type}
        return keypoints, score, exercise_type
    return keypoints, score


def preprocess_with_dict(path: tf.Tensor, score: tf.Tensor, exercise_type: tf.Tensor, training: bool = False):
    """Preprocess and return dict format for multi-output models."""
    keypoints, score_tensor, _ = preprocess(path, score, exercise_type, training=training, return_dict=False)
    return keypoints, {"score_output": score_tensor, "exercise_type_output": tf.cast(exercise_type, tf.int32)}


def build_tf_dataset(samples, training: bool, include_exercise_type: bool = False, model_has_multi_output: bool = False):
    """Build TensorFlow dataset from samples with optional exercise type."""
    if len(samples[0]) == 3:  # (path, score, exercise_type)
        paths, scores, exercise_types = zip(*samples)
        # Convert exercise types to indices
        exercise_to_idx = {ex: idx for idx, ex in enumerate(EXERCISE_TYPES)}
        exercise_indices = [exercise_to_idx.get(ex, len(EXERCISE_TYPES) - 1) for ex in exercise_types]
        
        if include_exercise_type:
            ds = tf.data.Dataset.from_tensor_slices((list(paths), list(scores), list(exercise_indices)))
            if training:
                ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
            if model_has_multi_output:
                # For multi-output models, return dict format
                ds = ds.map(lambda p, s, e: preprocess_with_dict(p, s, e, training=training), num_parallel_calls=tf.data.AUTOTUNE)
            else:
                ds = ds.map(lambda p, s, e: preprocess(p, s, e, training=training), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            ds = tf.data.Dataset.from_tensor_slices((list(paths), list(scores)))
            if training:
                ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
            ds = ds.map(lambda p, s: preprocess(p, s, training=training), num_parallel_calls=tf.data.AUTOTUNE)
    else:  # Backward compatibility: (path, score)
        paths, scores = zip(*samples)
        ds = tf.data.Dataset.from_tensor_slices((list(paths), list(scores)))
        if training:
            ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda p, s: preprocess(p, s, training=training), num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _in_dir(path_str: str, root: Path) -> bool:
    """Check if path is in directory."""
    try:
        Path(path_str).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ============================================================================
# Model Architecture
# ============================================================================

def build_model(high_accuracy: bool = True, num_exercise_types: int = 0) -> tf.keras.Model:
    """
    Build model optimized for highest accuracy with optional exercise type support.
    
    Args:
        high_accuracy: If True, uses larger model with more capacity
        num_exercise_types: If > 0, adds exercise type classification head
    
    Returns:
        Compiled Keras model
    """
    inputs = tf.keras.Input(shape=(NUM_FRAMES, FEATURE_DIMS))
    
    if high_accuracy:
        # MAXIMUM ACCURACY CONFIGURATION
        # Stacked BiLSTM layers for better temporal understanding
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name="bilstm_1"
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Second BiLSTM layer
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            name="bilstm_2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Attention mechanism for focusing on important frames
        # Self-attention (with fallback for older TF versions)
        try:
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=8, 
                key_dim=64,
                name="attention"
            )(x, x)
            x = tf.keras.layers.Add()([x, attention])  # Residual connection
            x = tf.keras.layers.LayerNormalization()(x)
        except (AttributeError, TypeError):
            # Fallback: Use simple attention mechanism
            # Compute attention weights
            attention_weights = tf.keras.layers.Dense(NUM_FRAMES, activation='softmax', name="attention_weights")(x)
            attention_weights = tf.expand_dims(attention_weights, axis=-1)
            attention = tf.reduce_sum(x * attention_weights, axis=1)
            x = tf.keras.layers.Add()([tf.reduce_mean(x, axis=1), attention])
            x = tf.keras.layers.LayerNormalization()(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Larger dense layers
        x = tf.keras.layers.Dense(
            512,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.5, name="dropout_1")(x)
        
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4, name="dropout_2")(x)
        
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_3"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_3")(x)
    else:
        # Standard configuration (faster training)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True),
            name="bilstm_1"
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        x = tf.keras.layers.Dense(
            256,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_1"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.4, name="dropout_1")(x)
        
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name="dense_2"
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3, name="dropout_2")(x)
    
    # Output layer for quality score
    score_output = tf.keras.layers.Dense(1, activation="sigmoid", name="score_output")(x)
    
    # Optional exercise type classification head
    if num_exercise_types > 0:
        exercise_output = tf.keras.layers.Dense(
            num_exercise_types,
            activation="softmax",
            name="exercise_type_output"
        )(x)
        outputs = {"score_output": score_output, "exercise_type_output": exercise_output}
    else:
        outputs = score_output
    
    return tf.keras.Model(inputs, outputs, name="multi_exercise_scorer")


# ============================================================================
# Training Functions
# ============================================================================

def plot_training_history(history, save_path: Path, has_exercise_types: bool = False):
    """Plot training history with support for multi-exercise models."""
    if not HAS_MATPLOTLIB:
        return
    
    # Determine number of subplots needed
    if has_exercise_types:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
    
    plot_idx = 0
    
    # Loss
    if 'loss' in history.history:
        axes[plot_idx].plot(history.history['loss'], label='Train Loss')
        if 'val_loss' in history.history:
            axes[plot_idx].plot(history.history['val_loss'], label='Val Loss')
        axes[plot_idx].set_title('Model Loss')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('Loss')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # MAE (handle both single and multi-output models)
    mae_key = 'score_output_mae' if has_exercise_types else 'mae'
    val_mae_key = 'val_score_output_mae' if has_exercise_types else 'val_mae'
    
    if mae_key in history.history:
        axes[plot_idx].plot(history.history[mae_key], label='Train MAE')
        if val_mae_key in history.history:
            axes[plot_idx].plot(history.history[val_mae_key], label='Val MAE')
        axes[plot_idx].set_title('Mean Absolute Error (Score)')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('MAE')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # RMSE
    rmse_key = 'score_output_rmse' if has_exercise_types else 'rmse'
    val_rmse_key = 'val_score_output_rmse' if has_exercise_types else 'val_rmse'
    
    if rmse_key in history.history:
        axes[plot_idx].plot(history.history[rmse_key], label='Train RMSE')
        if val_rmse_key in history.history:
            axes[plot_idx].plot(history.history[val_rmse_key], label='Val RMSE')
        axes[plot_idx].set_title('Root Mean Squared Error')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('RMSE')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # Exercise type accuracy (if multi-exercise)
    if has_exercise_types:
        if 'exercise_type_output_exercise_acc' in history.history:
            axes[plot_idx].plot(history.history['exercise_type_output_exercise_acc'], label='Train Acc')
            if 'val_exercise_type_output_exercise_acc' in history.history:
                axes[plot_idx].plot(history.history['val_exercise_type_output_exercise_acc'], label='Val Acc')
            axes[plot_idx].set_title('Exercise Type Accuracy')
            axes[plot_idx].set_xlabel('Epoch')
            axes[plot_idx].set_ylabel('Accuracy')
            axes[plot_idx].legend()
            axes[plot_idx].grid(True)
            plot_idx += 1
    
    # Learning Rate
    if 'lr' in history.history:
        axes[plot_idx].plot(history.history['lr'], label='Learning Rate')
        axes[plot_idx].set_title('Learning Rate Schedule')
        axes[plot_idx].set_xlabel('Epoch')
        axes[plot_idx].set_ylabel('LR')
        axes[plot_idx].set_yscale('log')
        axes[plot_idx].legend()
        axes[plot_idx].grid(True)
        plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    print(f"Training plots saved to {save_path}")
    plt.close()


def auto_detect_gpu():
    """
    Auto-detect and configure GPU (Mac MPS or NVIDIA CUDA).
    This is called automatically - no manual configuration needed.
    
    Returns:
        True if GPU is available, False otherwise
    """
    return setup_gpu()


def setup_gpu():
    """
    Configure TensorFlow to use GPU (NVIDIA CUDA or Mac MPS).
    
    Supports:
    - NVIDIA GPU with CUDA (Linux/Windows)
    - Mac GPU with Metal Performance Shaders (Apple Silicon: M1/M2/M3)
    
    Returns:
        True if GPU is available and configured, False otherwise
    """
    import platform
    
    is_mac = platform.system() == 'Darwin'
    is_apple_silicon = False
    
    # Check for Mac MPS (Metal Performance Shaders) - Apple Silicon
    if is_mac:
        try:
            # Check if running on Apple Silicon
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                  capture_output=True, text=True, timeout=2)
            cpu_brand = result.stdout.strip().lower()
            is_apple_silicon = 'apple' in cpu_brand or 'm1' in cpu_brand or 'm2' in cpu_brand or 'm3' in cpu_brand
            
            if is_apple_silicon:
                # Check for MPS devices (they show up as GPU on Mac with MPS support)
                gpus = tf.config.list_physical_devices('GPU')
                
                if gpus:
                    print(f"✓ Found Mac GPU (Metal Performance Shaders)")
                    print(f"  - Device: {gpus[0].name}")
                    print(f"  - CPU: {cpu_brand}")
                    print("  - Mac GPU acceleration enabled")
                    
                    # Verify MPS is functional
                    try:
                        with tf.device('/GPU:0'):
                            test_tensor = tf.constant([1.0, 2.0, 3.0])
                            _ = tf.reduce_sum(test_tensor)
                        print("  - MPS device verified and ready")
                        print("  - Note: MPS may be slower than CUDA for some operations")
                        print("  - For best performance, use appropriate batch sizes")
                        return True
                    except Exception as e:
                        print(f"  ⚠ MPS device found but not functional: {e}")
                        print("  - Falling back to CPU")
                        return False
                else:
                    print("⚠ Mac GPU (MPS) not available")
                    print("  - Ensure TensorFlow 2.5+ is installed")
                    print("  - MPS support is built into TensorFlow for Mac")
                    return False
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            # If we can't check, just proceed with normal GPU detection
            pass
    
    # Check for NVIDIA GPU (CUDA)
    gpus = tf.config.list_physical_devices('GPU')
    
    if not gpus:
        print("⚠ No GPU devices found. Training will use CPU (slower).")
        if is_mac and not is_apple_silicon:
            print("  Mac GPU (MPS) support:")
            print("    - Requires Apple Silicon (M1/M2/M3)")
            print("    - Requires TensorFlow 2.5+ with MPS support")
            print("    - MPS is automatically enabled if available")
        elif not is_mac:
            print("  To use NVIDIA GPU, ensure:")
            print("    1. NVIDIA GPU is installed")
            print("    2. CUDA toolkit is installed (https://developer.nvidia.com/cuda-downloads)")
            print("    3. cuDNN is installed (https://developer.nvidia.com/cudnn)")
            print("    4. tensorflow-gpu or tensorflow[and-cuda] is installed")
        return False
    
    # Configure NVIDIA GPU (if not Mac MPS)
    if not is_apple_silicon:
        print(f"✓ Found {len(gpus)} NVIDIA GPU(s)")
        
        try:
            # Enable memory growth to prevent TensorFlow from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  - {gpu.name}: Memory growth enabled")
            
            # Set mixed precision for faster training (optional but recommended)
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print("  - Mixed precision (float16) enabled for faster training")
            except Exception as e:
                print(f"  - Mixed precision not available: {e}")
            
            # Log GPU details
            for gpu in gpus:
                details = tf.config.experimental.get_device_details(gpu)
                print(f"  - GPU: {gpu.name}")
                if details:
                    print(f"    Compute Capability: {details.get('compute_capability', 'Unknown')}")
            
            return True
            
        except RuntimeError as e:
            # Memory growth must be set before GPUs are initialized
            print(f"⚠ GPU configuration error: {e}")
            print("  Continuing with default GPU settings...")
            return True
    
    return False


def main():
    parser = argparse.ArgumentParser(description="Train squat quality scoring model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--num-frames", type=int, default=24, help="Number of frames")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    parser.add_argument("--download-dataset", type=str, default=None, 
                       help="URL to download dataset from (optional)")
    parser.add_argument("--download-ucf101", action="store_true",
                       help="Download UCF-101 dataset using FiftyOne (requires Python 3.11+)")
    parser.add_argument("--download-coco", action="store_true",
                       help="Download COCO Pose dataset (requires manual download)")
    parser.add_argument("--process-coco", action="store_true",
                       help="Process downloaded COCO annotations")
    parser.add_argument("--setup-dirs", action="store_true",
                       help="Create data directories if they don't exist")
    parser.add_argument("--high-accuracy", action="store_true", default=True,
                       help="Use maximum accuracy model configuration (larger, slower)")
    parser.add_argument("--standard-model", action="store_true",
                       help="Use standard model (faster training, less accurate)")
    parser.add_argument("--cpu-only", action="store_true",
                       help="Force CPU usage (disable GPU)")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0,
                       help="Fraction of GPU memory to use (0.0-1.0, default: 1.0)")
    
    args = parser.parse_args()
    
    # Auto-detect and setup GPU before anything else
    if not args.cpu_only:
        print("="*60)
        print("GPU AUTO-DETECTION")
        print("="*60)
        gpu_available = auto_detect_gpu()  # Auto-detects Mac MPS or NVIDIA CUDA
        print("="*60 + "\n")
        
        if gpu_available:
            # Adjust batch size for GPU if needed
            if args.batch_size == 8:  # Default
                print("💡 Tip: With GPU, you can use larger batch sizes (e.g., --batch-size 16 or 32)")
                print("   This will speed up training significantly.\n")
    else:
        print("⚠ CPU-only mode enabled (GPU disabled)\n")
    
    # Update global config if provided
    global BATCH_SIZE, NUM_FRAMES
    BATCH_SIZE = args.batch_size
    NUM_FRAMES = args.num_frames
    
    print("="*60)
    print("WORKOUT QUALITY SCORING MODEL TRAINING")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    
    # Show GPU info
    import platform
    gpus = tf.config.list_physical_devices('GPU')
    if gpus and not args.cpu_only:
        is_mac = platform.system() == 'Darwin'
        if is_mac:
            print(f"GPU: Mac GPU (MPS) available")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print(f"GPU: {len(gpus)} NVIDIA device(s) available")
            for gpu in gpus:
                print(f"  - {gpu.name}")
    else:
        print("GPU: Not available (using CPU)")
    
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Number of frames: {NUM_FRAMES}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.epochs}")
    print(f"Early stopping patience: {args.patience}")
    print("="*60 + "\n")
    
    # Setup directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download UCF-101 dataset if requested
    if args.download_ucf101:
        print("Step 0: Downloading UCF-101 dataset (ALL WORKOUTS)...")
        if download_ucf101_all_workouts():
            print("✓ Dataset download complete! Please fill in scores in the CSV and run again.")
            return 0
        else:
            print("✗ Dataset download failed. Please check the error messages above.")
            return 1
    
    # Download COCO dataset if requested
    if args.download_coco:
        print("Step 0: Setting up COCO Pose dataset...")
        if download_coco_pose_dataset():
            print("✓ COCO dataset setup complete!")
            return 0
        else:
            print("✗ COCO dataset setup failed. Please check the error messages above.")
            return 1
    
    # Process COCO annotations if requested
    if args.process_coco:
        print("Step 0: Processing COCO annotations...")
        coco_dir = DATA_ROOT / "coco"
        if process_coco_annotations(coco_dir):
            print("✓ COCO annotations processed!")
            return 0
        else:
            print("✗ COCO annotation processing failed.")
            return 1
    
    # Download dataset if URL provided
    if args.download_dataset:
        print("Step 0: Downloading dataset from URL...")
        dataset_zip = MODEL_DIR / "dataset.zip"
        if download_from_url(args.download_dataset, dataset_zip):
            print("Dataset downloaded successfully!")
        else:
            print("Failed to download dataset. Please check the URL.")
            return 1
    
    # Setup directories if requested
    if args.setup_dirs:
        print("Step 0: Setting up data directories...")
        download_sample_dataset()
    
    # Check if data exists (including legacy directories)
    has_train_data = (
        (TRAIN_DIR.exists() and (any(TRAIN_DIR.rglob("*.mp4")) or any(TRAIN_DIR.rglob("*.avi")) or any(TRAIN_DIR.rglob("*.mov")))) or
        (OLD_TRAIN_DIR.exists() and (any(OLD_TRAIN_DIR.rglob("*.mp4")) or any(OLD_TRAIN_DIR.rglob("*.avi")) or any(OLD_TRAIN_DIR.rglob("*.mov"))))
    )
    has_test_data = (
        (TEST_DIR.exists() and (any(TEST_DIR.rglob("*.mp4")) or any(TEST_DIR.rglob("*.avi")) or any(TEST_DIR.rglob("*.mov")))) or
        (OLD_TEST_DIR.exists() and (any(OLD_TEST_DIR.rglob("*.mp4")) or any(OLD_TEST_DIR.rglob("*.avi")) or any(OLD_TEST_DIR.rglob("*.mov"))))
    )
    
    if not has_train_data and not args.setup_dirs:
        print("\n" + "="*60)
        print("WARNING: No training data found!")
        print("="*60)
        print(f"Expected videos in: {TRAIN_DIR}")
        print("\nOptions:")
        print("  1. Add videos to data/exercises_train/ and data/exercises_test/")
        print("  2. Run with --setup-dirs to create directories")
        print("  3. Run with --download-ucf101 to download UCF-101 dataset")
        print("  4. Run with --download-coco to setup COCO Pose dataset")
        print("  5. Run with --download-dataset <URL> to download a dataset")
        print("="*60 + "\n")
        response = input("Create directories now? (y/n): ").strip().lower()
        if response == 'y':
            download_sample_dataset()
        else:
            print("Please add your data and run again.")
            return 1
    
    # Prepare data
    print("Step 1: Preparing data...")
    ensure_label_file()
    labeled_samples = load_labeled_samples()
    
    if len(labeled_samples) < 2:
        raise ValueError("Add scores in the CSV (at least 2 labeled videos) before training.")
    
    # Split data (handle both old format (path, score) and new format (path, score, exercise_type))
    # Check both new and old directories
    train_samples = [
        s for s in labeled_samples 
        if _in_dir(s[0], TRAIN_DIR) or _in_dir(s[0], OLD_TRAIN_DIR)
    ]
    test_samples = [
        s for s in labeled_samples 
        if _in_dir(s[0], TEST_DIR) or _in_dir(s[0], OLD_TEST_DIR)
    ]
    
    if len(train_samples) < 2:
        raise ValueError("Need at least 2 labeled train videos in exercises_train for a train/val split.")
    
    random.shuffle(train_samples)
    split = max(1, int(0.8 * len(train_samples)))
    if split >= len(train_samples):
        split = len(train_samples) - 1
    
    print(f"Train samples: {len(train_samples[:split])}")
    print(f"Validation samples: {len(train_samples[split:])}")
    print(f"Test samples: {len(test_samples)}\n")
    
    # Check if exercise types are available
    has_exercise_types = len(train_samples[0]) == 3 if train_samples else False
    
    # Count unique exercise types in the dataset
    if has_exercise_types:
        unique_exercise_types = set(s[2] for s in train_samples + (test_samples if test_samples else []))
        num_unique_types = len(unique_exercise_types)
        # Only use multi-output if we have multiple exercise types
        if num_unique_types > 1:
            num_exercise_types = len(EXERCISE_TYPES)
        else:
            # Single exercise type - disable multi-output for simplicity
            num_exercise_types = 0
            has_exercise_types = False
            print(f"Note: Only one exercise type found ({unique_exercise_types.pop()}). Disabling multi-output mode.")
    else:
        num_exercise_types = 0
    
    # Build datasets
    print("Step 2: Building datasets...")
    # Determine if model will have multiple outputs
    model_has_multi_output = has_exercise_types and num_exercise_types > 0
    
    train_ds = build_tf_dataset(train_samples[:split], training=True, include_exercise_type=has_exercise_types, model_has_multi_output=model_has_multi_output)
    val_ds = build_tf_dataset(train_samples[split:], training=False, include_exercise_type=has_exercise_types, model_has_multi_output=model_has_multi_output)
    test_ds = build_tf_dataset(test_samples, training=False, include_exercise_type=has_exercise_types, model_has_multi_output=model_has_multi_output) if test_samples else None
    
    print(f"Train batches: {len(train_ds)}")
    print(f"Val batches: {len(val_ds)}")
    print(f"Test batches: {len(test_ds) if test_ds is not None else 0}\n")
    
    # Build model
    print("Step 3: Building model...")
    use_high_accuracy = args.high_accuracy and not args.standard_model
    if use_high_accuracy:
        print("Using MAXIMUM ACCURACY configuration:")
        print("  - Stacked BiLSTM (512→256 units)")
        print("  - Multi-head attention mechanism")
        print("  - Larger dense layers (512→256→128)")
        print("  - More regularization")
    else:
        print("Using STANDARD configuration (faster training)")
    
    if has_exercise_types:
        print(f"  - Multi-exercise support: {num_exercise_types} exercise types")
    
    model = build_model(high_accuracy=use_high_accuracy, num_exercise_types=num_exercise_types)
    
    # Use lower learning rate for high-accuracy model
    lr = args.learning_rate if not use_high_accuracy else args.learning_rate * 0.5
    
    # Configure loss and metrics based on model outputs
    if has_exercise_types and num_exercise_types > 0:
        loss = {
            "score_output": tf.keras.losses.Huber(delta=1.0),
            "exercise_type_output": tf.keras.losses.SparseCategoricalCrossentropy()
        }
        loss_weights = {"score_output": 1.0, "exercise_type_output": 0.3}
        metrics = {
            "score_output": [
                tf.keras.metrics.MeanAbsoluteError(name="mae"),
                tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            ],
            "exercise_type_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="exercise_acc")]
        }
    else:
        loss = tf.keras.losses.Huber(delta=1.0)
        loss_weights = None
        metrics = [
            tf.keras.metrics.MeanAbsoluteError(name="mae"),
            tf.keras.metrics.RootMeanSquaredError(name="rmse"),
            tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
        ]
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.summary()
    print()
    
    # Setup callbacks
    print("Step 4: Setting up callbacks...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=args.patience,
            restore_best_weights=True,
            monitor="val_mae",
            verbose=1,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "model.keras"),
            save_best_only=True,
            monitor="val_mae",
            verbose=1,
            save_weights_only=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
            mode="min"
        ),
    ]
    print("Callbacks configured.\n")
    
    # Train
    print("Step 5: Training model...")
    print("="*60)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    print("="*60 + "\n")
    
    # Training summary
    if has_exercise_types and num_exercise_types > 0:
        val_mae_key = "val_score_output_mae"
        if val_mae_key not in history.history:
            val_mae_key = "val_mae"  # Fallback
    else:
        val_mae_key = "val_mae"
    
    if val_mae_key in history.history:
        best_val_mae = min(history.history[val_mae_key])
        best_epoch = history.history[val_mae_key].index(best_val_mae) + 1
    else:
        best_val_mae = history.history.get("val_loss", [0])[-1]
        best_epoch = len(history.history.get("val_loss", []))
    
    print("="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Best validation MAE (normalized 0-1): {best_val_mae:.6f}")
    print(f"Best validation MAE (score units): {best_val_mae * SCORE_SCALE:.2f}")
    print(f"Best epoch: {best_epoch}/{len(history.history.get('loss', []))}")
    print(f"Final learning rate: {model.optimizer.learning_rate.numpy():.2e}")
    if has_exercise_types and "val_exercise_type_output_exercise_acc" in history.history:
        best_ex_acc = max(history.history["val_exercise_type_output_exercise_acc"])
        print(f"Best exercise type accuracy: {best_ex_acc:.4f}")
    print("="*60 + "\n")
    
    # Evaluate
    print("Step 6: Evaluating on test set...")
    eval_target = test_ds if test_ds is not None else val_ds
    eval_results = model.evaluate(eval_target, return_dict=True, verbose=1)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    for metric, value in eval_results.items():
        if 'mae' in metric.lower():
            print(f"{metric.upper()}: {value:.6f} (normalized) = {value * SCORE_SCALE:.2f} (score units)")
        elif 'acc' in metric.lower():
            print(f"{metric.upper()}: {value:.4f} ({value*100:.2f}%)")
        else:
            print(f"{metric.upper()}: {value:.6f}")
    print("="*60 + "\n")
    
    # Save model
    print("Step 7: Saving model...")
    export_dir = MODEL_DIR / "squat_scorer.keras"
    model.save(export_dir)
    with (MODEL_DIR / "score_scale.txt").open("w", encoding="utf-8") as f:
        f.write(str(SCORE_SCALE))
    
    print(f"Model saved to {export_dir}")
    print("Artifacts saved to", MODEL_DIR)
    
    # Plot history
    if HAS_MATPLOTLIB:
        plot_training_history(history, MODEL_DIR / "training_history.png", has_exercise_types=has_exercise_types)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())


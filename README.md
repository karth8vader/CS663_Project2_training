# Exercise Quality Scoring Model - Complete Guide

This guide covers the complete workflow for downloading datasets, training, testing, and deploying the exercise quality scoring model.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Downloading Datasets](#downloading-datasets)
4. [Preparing Training Data](#preparing-training-data)
5. [Training the Model](#training-the-model)
6. [Testing the Model](#testing-the-model)
7. [Converting to TFLite](#converting-to-tflite)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

- **Python**: 3.8+ (Python 3.11+ recommended for FiftyOne/UCF-101)
- **Operating System**: Linux, macOS, or Windows
- **GPU** (optional but recommended):
  - NVIDIA GPU with CUDA (Linux/Windows)
  - Apple Silicon (M1/M2/M3) with Metal Performance Shaders (macOS)
- **Disk Space**: 
  - Minimum: ~5GB for basic setup
  - Recommended: ~60GB for full UCF-101 dataset

### GPU Setup

#### NVIDIA GPU (Linux/Windows)
1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
2. Install cuDNN: https://developer.nvidia.com/cudnn
3. Install TensorFlow with CUDA support:
   ```bash
   pip install tensorflow[and-cuda]
   ```

#### Mac GPU (Apple Silicon)
- No additional setup needed! MPS is automatically enabled in TensorFlow 2.5+
- Just install TensorFlow normally

---

## Installation

### 1. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
# Check TensorFlow installation
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Check GPU availability
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### 3. Create Directory Structure

The scripts will create directories automatically, but you can also create them manually:

```bash
mkdir -p data/exercises_train
mkdir -p data/exercises_test
mkdir -p checkpoints
```

---

## Downloading Datasets

### Option 1: Download UCF-101 Dataset (Recommended)

The UCF-101 dataset contains multiple exercise types and is ideal for training.

**Requirements:**
- Python 3.11+ (for FiftyOne)
- ~50GB free disk space
- Stable internet connection

**Steps:**

```bash
# Download UCF-101 dataset (all workout classes)
python download_datasets.py --ucf101

# Or use the training script directly
python train_model.py --download-ucf101
```

**What it does:**
- Downloads UCF-101 dataset (~50GB)
- Extracts exercise-related videos (squats, pushups, pullups, etc.)
- Splits into train/test (80/20)
- Saves to `data/squats_train/` and `data/squats_test/`

**Note:** The download may take several hours depending on your internet speed.

### Option 2: Download COCO Pose Dataset

COCO provides diverse human pose annotations suitable for exercise recognition.

```bash
# Download COCO annotations
python download_datasets.py --coco

# Or use the training script
python train_model.py --download-coco
```

**After downloading annotations:**
1. Manually download images from: https://cocodataset.org/#download
   - `train2017.zip` (~18GB)
   - `val2017.zip` (~1GB)
2. Extract to `data/coco/train2017/` and `data/coco/val2017/`
3. Process annotations:
   ```bash
   python train_model.py --process-coco
   ```

### Option 3: Use Your Own Videos

1. Place your video files in the training directories:
   ```bash
   # Training videos
   data/exercises_train/
     ├── video1.mp4
     ├── video2.mp4
     └── ...
   
   # Test videos
   data/exercises_test/
     ├── test1.mp4
     └── ...
   ```

2. Supported formats: `.mp4`, `.mov`, `.avi`, `.mkv`

### List Available Datasets

```bash
python download_datasets.py --list
```

---

## Preparing Training Data

### 1. Generate Label CSV

After adding videos, generate the label file:

```bash
python train_model.py
```

This creates `data/exercise_scores.csv` with columns:
- `relative_path`: Path to video file
- `exercise_type`: Type of exercise (squat, pushup, etc.)
- `score`: Quality score (0-100) - **you need to fill this in**

### 2. Fill in Scores

Open `data/exercise_scores.csv` and fill in:
- **score**: Quality score from 0-100 (0 = poor form, 100 = perfect form)
- **exercise_type**: One of the supported types:
  - `squat`, `pushup`, `pullup`, `lunge`, `plank`, `burpee`
  - `jumping_jack`, `mountain_climber`, `situp`, `deadlift`
  - `bench_press`, `dumbbell_curl`, `overhead_press`, `other`

**Example CSV:**
```csv
relative_path,exercise_type,score
exercises_train/squat_001.mp4,squat,85
exercises_train/pushup_001.mp4,pushup,72
exercises_test/squat_test.mp4,squat,90
```

### 3. Minimum Data Requirements

- **Minimum**: 2 labeled videos (for basic testing)
- **Recommended**: 50+ videos per exercise type
- **Ideal**: 200+ videos with diverse quality scores

---

## Training the Model

### Basic Training

```bash
# Train with default settings
python train_model.py
```

**Default settings:**
- Epochs: 100
- Batch size: 8
- Learning rate: 1e-4
- Early stopping patience: 7
- High accuracy model configuration

### Advanced Training Options

```bash
# Custom training parameters
python train_model.py \
    --epochs 150 \
    --batch-size 16 \
    --learning-rate 1e-4 \
    --patience 10 \
    --high-accuracy

# Use standard (faster) model
python train_model.py --standard-model

# Force CPU-only training
python train_model.py --cpu-only

# Custom number of frames
python train_model.py --num-frames 32
```

### Training Options Explained

| Option | Description | Default |
|--------|-------------|---------|
| `--epochs` | Number of training epochs | 100 |
| `--batch-size` | Batch size (increase for GPU) | 8 |
| `--learning-rate` | Initial learning rate | 1e-4 |
| `--patience` | Early stopping patience | 7 |
| `--high-accuracy` | Use maximum accuracy model | Enabled |
| `--standard-model` | Use faster standard model | Disabled |
| `--cpu-only` | Force CPU usage | Disabled |
| `--num-frames` | Number of frames per video | 24 |

### Training Output

The training script will:
1. **Extract keypoints** from videos (cached for speed)
2. **Build the model** (BiLSTM-based architecture)
3. **Train** with validation monitoring
4. **Save best model** to `checkpoints/squat_scorer.keras`
5. **Generate plots** (if matplotlib available) to `checkpoints/training_history.png`

### Model Architecture

**High Accuracy Configuration:**
- Stacked BiLSTM layers (512→256 units)
- Multi-head attention mechanism
- Dense layers (512→256→128)
- Dropout and batch normalization
- Multi-exercise support (if multiple exercise types in data)

**Standard Configuration:**
- Single BiLSTM layer (256 units)
- Simpler architecture
- Faster training, slightly lower accuracy

### Monitoring Training

The script displays:
- Training/validation loss
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Exercise type accuracy (if multi-exercise)
- Learning rate schedule

**Best model** is automatically saved based on validation MAE.

---

## Testing the Model

### Test on Video File

```bash
# Test with default model
python test_model.py path/to/video.mp4

# Test with specific model
python test_model.py path/to/video.mp4 --model checkpoints/squat_scorer.keras

# Test TFLite model
python test_model.py path/to/video.mp4 --model model.tflite

# Disable keypoint caching
python test_model.py path/to/video.mp4 --no-cache
```

**Output:**
```
==================================================
Predicted Quality Score: 85.23 / 100
Detected Exercise: Squat
Confidence: 92.5%
==================================================
```

### Test on Live Camera Stream

```bash
# Use default camera (camera 0)
python test_model.py --live

# Use specific camera
python test_model.py --live --camera 1

# Use specific model
python test_model.py --live --model checkpoints/squat_scorer.keras
```

**Live Stream Features:**
- Real-time pose detection and visualization
- Quality score display (color-coded: green ≥70, orange ≥50, red <50)
- Exercise type detection (if multi-exercise model)
- FPS counter
- Press 'q' to quit

### Test Options

| Option | Description |
|--------|-------------|
| `video_path` | Path to video file (required if not using --live) |
| `--model` | Path to model file (.keras or .tflite) | `checkpoints/squat_scorer.keras` |
| `--live` | Use live camera stream instead of video |
| `--camera` | Camera device ID for live stream | 0 |
| `--no-cache` | Disable keypoint caching | Disabled |

### Understanding Scores

- **Score Range**: 0-100
  - 90-100: Excellent form
  - 70-89: Good form
  - 50-69: Acceptable form
  - 0-49: Poor form (needs improvement)

- **Exercise Type**: Detected exercise (if multi-exercise model)
- **Confidence**: Model confidence in exercise type detection (0-100%)

---

## Converting to TFLite

### Basic Conversion

```bash
# Convert default model
python convert_to_tflite.py

# Convert specific model
python convert_to_tflite.py --model checkpoints/squat_scorer.keras --output model.tflite
```

### Conversion Options

```bash
# Disable optimizations (for debugging)
python convert_to_tflite.py --no-optimize

# Apply INT8 quantization (smaller file, may reduce accuracy)
python convert_to_tflite.py --quantize

# Test converted model
python convert_to_tflite.py --test
```

### Conversion Options Explained

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | Input Keras model path | `model.keras` |
| `--output` | Output TFLite file path | `squat_scorer.tflite` |
| `--no-optimize` | Disable optimizations | Disabled |
| `--quantize` | Apply INT8 quantization | Disabled |
| `--test` | Test converted model | Disabled |

### Conversion Process

1. **Load Keras model** from `.keras` file
2. **Export to SavedModel** format
3. **Convert to TFLite** with optimizations
4. **Save** `.tflite` file

**Note:** LSTM/BiLSTM models require `SELECT_TF_OPS` for TFLite. The converter handles this automatically.

### Using TFLite Model

The TFLite model can be used in:
- **Android apps** (with TensorFlow Lite)
- **iOS apps** (with TensorFlow Lite)
- **Edge devices** (Raspberry Pi, etc.)
- **Python** (using `test_model.py`)

**For Android:**
Add to `build.gradle`:
```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.x.x'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:2.x.x'
}
```

---

## Troubleshooting

### Common Issues

#### 1. "No training data found"

**Solution:**
```bash
# Create directories and add videos
python train_model.py --setup-dirs

# Or manually create:
mkdir -p data/exercises_train data/exercises_test
# Then add your video files
```

#### 2. "Keras dependencies not installed"

**Solution:**
```bash
# Install TensorFlow/Keras
pip install "tensorflow<2.17"

# Or install all requirements
pip install -r requirements.txt
```

#### 3. "GPU not detected"

**For NVIDIA GPU:**
- Verify CUDA installation: `nvidia-smi`
- Install TensorFlow with CUDA: `pip install tensorflow[and-cuda]`
- Check TensorFlow GPU: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

**For Mac GPU:**
- Ensure you have Apple Silicon (M1/M2/M3)
- TensorFlow 2.5+ automatically enables MPS
- No additional setup needed

**Force CPU:**
```bash
python train_model.py --cpu-only
```

#### 4. "Out of memory" during training

**Solutions:**
- Reduce batch size: `--batch-size 4`
- Use standard model: `--standard-model`
- Reduce number of frames: `--num-frames 16`

#### 5. "TFLite conversion failed"

**Solutions:**
- Ensure model is saved correctly: `model.save('model.keras')`
- Try without optimizations: `--no-optimize`
- Check TensorFlow version: `pip install "tensorflow<2.17"`

#### 6. "FiftyOne not available" (UCF-101 download)

**Solution:**
```bash
# Install FiftyOne (requires Python 3.11+)
pip install fiftyone

# Or use your own videos instead
```

#### 7. "Video file not found" during testing

**Solution:**
- Check video path is correct
- Ensure video format is supported (`.mp4`, `.mov`, `.avi`, `.mkv`)
- Check file permissions

#### 8. "No pose detected" in video

**Solutions:**
- Ensure person is clearly visible in frame
- Check lighting conditions
- Try different video angle
- MediaPipe requires person to be facing camera

### Performance Tips

1. **GPU Training:**
   - Use larger batch sizes (16-32) for faster training
   - Enable mixed precision automatically

2. **Keypoint Caching:**
   - Keypoints are cached automatically (saves time on re-runs)
   - Clear cache: Delete `checkpoints/keypoints_cache/`

3. **Model Size:**
   - Standard model: Faster training, smaller file
   - High accuracy model: Better results, larger file

4. **TFLite Optimization:**
   - Use quantization for smaller file size
   - Trade-off: Slight accuracy reduction

---

## File Structure

After setup, your directory should look like:

```
.
├── data/
│   ├── exercises_train/          # Training videos
│   ├── exercises_test/            # Test videos
│   ├── exercise_scores.csv        # Labels (fill in scores)
│   └── coco/                      # COCO dataset (if downloaded)
├── checkpoints/
│   ├── squat_scorer.keras         # Trained model
│   ├── model.keras                # Best model during training
│   ├── training_history.png       # Training plots
│   └── keypoints_cache/           # Cached keypoints
├── test_model.py                  # Testing script
├── train_model.py                 # Training script
├── download_datasets.py           # Dataset downloader
├── convert_to_tflite.py          # TFLite converter
├── requirements.txt               # Dependencies
└── README.md                      # This file
```

---

## Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (or use your own videos)
python download_datasets.py --ucf101

# 3. Prepare labels
python train_model.py  # Creates CSV, fill in scores

# 4. Train model
python train_model.py --epochs 100

# 5. Test model
python test_model.py video.mp4
python test_model.py --live

# 6. Convert to TFLite (optional)
python convert_to_tflite.py --model checkpoints/squat_scorer.keras
```

---

## Additional Resources

- **TensorFlow Documentation**: https://www.tensorflow.org/
- **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose
- **TFLite Android Guide**: https://www.tensorflow.org/lite/android/ops_select
- **UCF-101 Dataset**: https://www.crcv.ucf.edu/data/UCF101.php

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review script help: `python <script>.py --help`
3. Check error messages and stack traces

---

**Happy Training! 🏋️‍♂️**


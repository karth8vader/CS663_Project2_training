This repository is a university-style project for training and evaluating lightweight exercise-quality scoring models. The work is intentionally compact and reproducible, aimed at teaching practical end-to-end ML pipelines for human pose based regression.

## Project scope and purpose

-   Objective: train small neural regressors that predict a quality score for input videos of exercise repetitions (squat, push-up, lunge).
-   Intended audience: students and researchers who want a minimal, reproducible example of a pose-based ML pipeline (preprocessing, model training, evaluation, and export).

## Core approach (high-level)

-   Pose extraction: MediaPipe BlazePose (33 landmarks per frame). Each landmark contains (x, y, z, visibility).
-   Temporal sampling: videos are uniformly sampled to NUM_FRAMES timesteps (default 16) and per-frame landmarks are normalized and rotated to a canonical pose-centered coordinate frame.
-   Feature layout: per-timestep flattened landmarks plus a per-frame validity flag — final input shape for the model is (NUM_FRAMES, NUM_LANDMARKS \* LANDMARK_DIMS + 1).
-   Model: a lightweight Bi-directional LSTM (128 units per direction) followed by global pooling and two dense layers. Final output is a sigmoid-normalized scalar (0..1) which is scaled to human-readable scores (0..100).

## Technical details & constants (matching the notebooks)

-   NUM_FRAMES = 16 (temporal sequence length)
-   NUM_LANDMARKS = 33 (BlazePose full-body landmarks)
-   LANDMARK_DIMS = 4 (x, y, z, visibility)
-   SCORE_SCALE = 100.0 (labels are recorded on a 0..100 scale; the model trains on 0..1)
-   Input features per timestep = NUM_LANDMARKS \* LANDMARK_DIMS + 1 (the extra feature is a per-frame validity flag)

## Typical model architecture

-   Input: (NUM_FRAMES, FEATURE_DIMS)
-   BiDirectional(LSTM(128, return_sequences=True))
-   GlobalAveragePooling1D
-   Dense(128, relu) -> Dropout(0.3) -> Dense(64, relu)
-   Output: Dense(1, sigmoid) — scaled to 0..100 for reporting

## Training & evaluation

-   Loss: mean-squared error (MSE) on normalized target (0..1).
-   Metric: mean absolute error (MAE). Notation: MAE reported is on model normalization (0..1) and usually also shown in score units (MAE \* SCORE_SCALE).
-   Callbacks: early stopping + ModelCheckpoint (best checkpoint saved to `checkpoints/model.keras`).

## Files and layout

-   notebooks: `squat_training.ipynb`, `pushup_training.ipynb`, `lunge_training.ipynb` — each is an end-to-end training example.
-   data/: local dataset layout expected as:
    -   data/squats_train, data/squats_test
    -   data/PushUps_train, data/PushUps_test
    -   data/lunges_train, data/lunges_test
    -   per-exercise CSV files mapping video path -> score (e.g., `data/squat_scores.csv`).
-   checkpoints/: training artifacts and exports. Typical files produced by the notebooks:
    -   model.keras — checkpoint saved during training (best val MAE)
    -   <exercise>\_scorer.keras — single-file Keras archive (model export)
    -   <exercise>\_scorer_savedmodel/ — SavedModel directory (export fallback for portability)
    -   score_scale.txt — records SCORE_SCALE used when saving artifacts

## Preprocessing & reproducibility notes

-   Preprocessing runs Pose detection (MediaPipe BlazePose) per sampled frame, then applies a visibility-aware center/scale/rotation normalization and temporal interpolation. This preprocessing is implemented in Python helpers shared across notebooks and is intentionally run outside the TF graph (via tf.py_function when used inside the tf.data pipeline).
-   Keypoint outputs are cached under `checkpoints/keypoints_cache/` (sha1 of path) so repeated experiments are faster.
-   For reproducibility, each notebook seeds Python + NumPy + TensorFlow RNGs where appropriate.

## Scoring and evaluation tools

-   `scripts/score_video.py` — a CLI that mirrors the notebook preprocessing and can score single video files using any exported model in `checkpoints/`.
-   `scripts/test_models.sh` — a convenience Bash helper that runs `score_video.py` on up to five test videos per exercise directory (use Git Bash / WSL on Windows, or run the PowerShell equivalent shown in the script header comments).

## Notes and limitations (important for students)

-   These models are intentionally small and designed for instructional use — they are not production-grade. LSTM-based BiLSTM models may not be optimal for on-device inference and can require additional engineering (model simplification, conversion strategies, or different runtimes) to deploy.
-   This project focuses on training, evaluation, and reproducible experimentation rather than deployment — the repo contains helper scripts and example flows for exporting models for downstream use, but conversion/export paths may require environment-specific dependencies.

## If you want further work:

-   Replace LSTM with more TFLite-friendly alternatives (e.g., temporal convolution or GRU) if the goal is smaller on-device models.
-   Add automated data validation and unit tests for preprocessing to ensure long-term reproducibility across research runs.

If you'd like, I can add a short 'how-to-run' section for new students (conda env, required packages) or add a small PowerShell test script that mirrors `scripts/test_models.sh`.

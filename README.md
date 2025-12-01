This repo trains lightweight BlazePose+BiLSTM models for exercise quality scoring (squat/pushup/lunge).

Artifacts and conversion workflow

-   Training notebooks now export two artifacts under `checkpoints/`:

    -   `<exercise>_scorer.keras` — single-file Keras archive (native Keras format)
    -   `<exercise>_scorer_savedmodel/` — SavedModel directory (preferred for TFLite conversion)

-   Converting to TFLite (recommended):
    -   Use the included CLI `scripts/convert_to_tflite.py` and point to either the `.keras` archive or the `.keras` file path; the script will prefer a matching `*_savedmodel` directory if present and will use TF Select (SELECT_TF_OPS) when required for LSTM ops.

Example:

```powershell
# Convert using the SavedModel directory (preferred)
python scripts/convert_to_tflite.py --src checkpoints/squat_scorer_savedmodel --out checkpoints/squat_scorer.tflite

# Or point to the .keras file — script will detect a corresponding savedmodel dir if available
python scripts/convert_to_tflite.py --src checkpoints/squat_scorer.keras --out checkpoints/squat_scorer.tflite
```

Notes:

-   LSTM/Bidirectional models often require SELECT_TF_OPS (TF Select) and may produce TFLite artifacts that require a TF runtime / Flex delegate.
-   For pure TFLite models without TF Select ops, consider reworking RNN layers to TFLite-friendly alternatives (e.g., GRU or simpler temporal pooling) and re-export.

# BirdCLEF 2026 Example Solution

This repository codifies a BirdCLEF 2026-style baseline: start from the Kaggle-provided cleaned clips, convert them into log-mel spectrograms, train a lightweight CNN, and reuse the same feature extractor during inference to score the long-form soundscapes.

## Data layout
1. `train_metadata.csv` – each row describes a short clip along with the `primary_label`, optional `secondary_labels`, and its filename inside `train_audio/`.
2. `train_audio/` – 5-10 second recordings grouped by species (the script will search for each `filename` under this folder and optionally under `train_audio/<label>/`).
3. `train_soundscapes/` – unlabeled long recordings for pseudo-labeling or validation helpers.
4. `test_soundscapes.csv` and accompanying `test_soundscapes/` – the public and private leaderboard predictions are produced on these minute-long files.
5. `taxonomy.csv` – contains the allowed species codes; `build_label_map` fetches `primary_label`, `ebird_code`, or `species_id` from it to keep consistent ordering across splits.

Place those folders under `data/` (or anywhere you like) and pass their paths to the CLI scripts below.

## Training
Run a single-GPU (or CPU) training job that automatically splits the metadata:

```
python -m birdclef_example.train \
  --audio-dir data/train_audio \
  --metadata data/train_metadata.csv \
  --taxonomy data/taxonomy.csv \
  --output-dir outputs \
  --epochs 15 \
  --batch-size 48 \
  --segment-duration 5.0
```

The script resamples every clip to 32 kHz, crops/pads a 5-second window, converts it to a normalized log-mel spectrogram, and minimizes BCE-with-logits against the multi-hot labels. The best checkpoint is stored in `outputs/best_model.pt` along with `label_map.json` for prediction.

## Inference & submission
Score the minute-long soundscapes by chopping them into overlapping 5-second windows, running the CNN in batches, and averaging the sigmoid probabilities. The CLI writes a submission with one row per soundscape (`row_id`) and one column per species:

```
python -m birdclef_example.predict \
  --model-path outputs/best_model.pt \
  --label-map outputs/label_map.json \
  --soundscape-dir data/test_soundscapes \
  --metadata data/test_soundscapes.csv \
  --output-csv outputs/submission.csv
```

Kaggle expects a wide CSV (one column per species code) where each cell holds the probability that the species appears in the corresponding five-second window.

## Evaluation & expectations
BirdCLEF continues to use a macro-averaged ROC-AUC that ignores classes never seen in the test fold. Each `row_id` therefore represents a fixed-time window and the submission file includes one probability per species, so the leaderboard reward is based on ranking rather than hard thresholds. The 2025 edition also restricted inference to under two hours on a single CPU, so prioritizing fast spectrogram extraction and batching strategies helps keep production runs in-budget.

## Extensions
- Swap `SimpleCNN` for a `timm` backbone or EfficientNet-like encoder and adjust the dropout/learning-rate schedule.
- Add pseudo-labeling by scoring `train_soundscapes/` and re-training on confident predictions.
- Use `torchaudio`'s `FrequencyMasking`/`TimeMasking` or `SpecAugment` to improve generalization.

Refer to the Kaggle competition page for the latest rules, private-test constraints, and entry dates.

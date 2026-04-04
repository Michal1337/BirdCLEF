# BirdCLEF Research Program

This project is for iterative BirdCLEF experimentation with clear guardrails.

The goal is simple:
- Improve validation `BirdCLEF ROC AUC`.

The main constraint is also simple:
- Keep wall clock time in mind. Do not mindlessly increase model size or training cost for tiny or speculative gains.

Execution rule:
- The agent should only make code changes and update logs or notes.
- The user runs the actual training command manually.
- The agent must not launch training runs on its own unless the user explicitly asks for that in a separate instruction.

## Scope

You may modify the training stack, including:
- `birdclef_example/train.py`
- `birdclef_example/model.py`
- `birdclef_example/data.py`
- other training-support files in `birdclef_example/` if needed

You must not modify the evaluation definition itself.
- Do not change the validation split policy unless explicitly requested.
- Do not change the BirdCLEF ROC AUC computation logic to make results look better.
- Do not redefine success metrics away from BirdCLEF ROC AUC.

Refactors that make evaluation code cleaner are fine only if they preserve behavior exactly.

## Setup

Before experimentation:
1. Read the current training stack for full context:
   - `birdclef_example/train.py`
   - `birdclef_example/model.py`
   - `birdclef_example/data.py`
   - `birdclef_example/utils.py`
2. Confirm the dataset paths exist:
   - `data/train.csv`
   - `data/train_soundscapes_labels.csv`
   - `data/train_audio/`
   - `data/train_soundscapes/`
   - `data/taxonomy.csv`
3. Use the canonical train command from repo root:
   - `python -m birdclef_example.train`
4. Keep a plain text or markdown log file with:
   - experiment idea
   - reasoning
   - key config changes
   - observed BirdCLEF ROC AUC
   - brief keep/discard decision

Suggested filename:
- `experiment_log.md`

If the log file does not exist, create it before starting experiments.

## Optimization Target

Primary target:
- Highest validation `BirdCLEF ROC AUC`

Secondary judgment criteria:
- similar or better score with less complexity is a win
- similar or better score with lower memory or shorter training is a win
- tiny metric gains that require much larger models, much slower training, or brittle code are usually not worth keeping

The metric is the decision-maker.
- Prefer changes that genuinely improve ROC AUC.
- Do not optimize for train loss, validation loss, parameter count, or throughput unless they help BirdCLEF ROC AUC.

## Wall Clock Discipline

This project does not use a fixed hard-coded 5 minute budget, but wall clock time matters.

Guidelines:
- Prefer experiments that are reasonably comparable in runtime.
- Avoid large architecture jumps without a strong reason.
- Be skeptical of changes that substantially increase compute, memory, or startup cost.
- If a run is clearly too slow, too memory-heavy, or impractical, record that and move on.
- Favor ideas that improve quality per unit of compute, not just absolute size.

Good changes:
- stronger inductive bias
- better pooling or sequence modeling
- more stable optimization
- better augmentations or sampling
- cleaner use of available GPU features

Bad changes:
- blindly doubling width/depth repeatedly
- adding complexity without a clear hypothesis
- making the pipeline fragile or hard to debug

## What To Try

Reasonable experiment directions:
- model architecture improvements
- optimizer and scheduler tuning
- batch size and accumulation changes
- precision and throughput improvements
- better caching / input pipeline efficiency
- regularization and dropout tuning
- augmentation changes
- soundscape / clip sampling strategy improvements
- sequence aggregation or transformer design changes

Keep evaluation behavior fixed.

## What Not To Do

Do not:
- redefine the validation metric
- alter the validation split just to get a better number
- add hacks that leak validation information into training
- optimize solely for model size growth
- install random new dependencies unless explicitly approved
- let logs flood the terminal unnecessarily during long experiment loops

## Experiment Loop

For each experiment:
1. Start from the current best known working state.
2. State the hypothesis in one or two sentences in the log file.
3. Make the smallest code changes that test that hypothesis.
4. Tell the user what command to run:
   - `torchrun --standalone --nproc_per_node=2 -m birdclef_example.train_ddp`
5. After the user runs it, record:
   - BirdCLEF ROC AUC
   - notable runtime or memory observations
   - whether the idea seems worth keeping
6. Keep the change only if it is a meaningful improvement overall.
7. If the result is worse, noisy, unstable, or too expensive for the gain, revert and try something else.

If a run crashes:
1. Record the idea and failure in the log.
2. Note whether it was:
   - bug
   - OOM
   - numerical instability
   - impractical runtime
3. Fix obvious implementation mistakes if the core idea is still good.
4. Otherwise discard it and move on.

The agent should prepare the code for the next run and clearly hand off the command to the user.

## Logging Format

Keep a running file with entries like:

```md
## 2026-03-28 Exp 01
Idea: Replace CNN head with compact spectrogram transformer.
Reasoning: Better temporal modeling may improve multilabel soundscape discrimination.
Changes: `model.py`, `train.py`
Metric: val_birdclef_roc_auc=0.8123
Runtime Notes: Slightly slower per epoch, still acceptable.
Decision: keep
```

Another example:

```md
## 2026-03-28 Exp 02
Idea: Double transformer width.
Reasoning: Test whether capacity is the bottleneck.
Changes: `model.py`
Metric: val_birdclef_roc_auc=0.8118
Runtime Notes: Noticeably slower and heavier VRAM use.
Decision: discard
```

The log should make it easy to answer:
- what was tried
- why it was tried
- what metric it achieved
- whether it was worth keeping

## Simplicity Rule

All else equal, prefer:
- simpler code
- clearer reasoning
- smaller diffs
- faster iteration
- stable training

A small ROC AUC gain from a clean improvement is valuable.
A tiny ROC AUC gain from a messy, expensive, brittle change is usually not.

## Success Condition

A successful experiment program:
- improves validation BirdCLEF ROC AUC
- keeps the evaluation honest
- respects wall clock and memory
- leaves behind a clear written record of ideas, reasoning, and outcomes

# Pretrain CLI refactor report

## Modified files

- `fine_tune.py`
  - Added command-line control for whether fine-tuning should use pretrained encoder weights.
  - Added per-run output files under `runs/{run_name}/`.
  - Added dry-run mode for checking argument resolution, model initialization, and pretrained checkpoint loading without training.
  - Added pretrain loading audit output.

`models/Encoder.py` was not changed. The existing `VIT_GIN_Parallel(pretrained_path=...)` and `load_pretrained_encoder()` logic is still used.

## New command-line arguments

- `--use_pretrain`
  - Default: `false`
  - Controls whether `fine_tune.py` passes a checkpoint path into `VIT_GIN_Parallel`.

- `--pretrained_path`
  - Default: `None`
  - Path to the pretrained checkpoint, for example `weights/pre_model.pt`.
  - Required only when `--use_pretrain true`.
  - If `--use_pretrain true` and the file does not exist, the script stops with `PRETRAIN_CHECKPOINT_NOT_FOUND`.

- `--run_name`
  - Default: auto-generated timestamp name.
  - Controls the run folder: `runs/{run_name}/`.

- `--dry_run`
  - Default: `false`
  - When true, the script only parses arguments, creates the run folder, writes resolved config, initializes the model, checks pretrained loading, writes the audit file, and exits without training.

- `--skip_test`
  - Default: `true`
  - When true, the training loop skips the per-epoch test-set evaluation.

## Default behavior

By default, pretraining is disabled:

```bash
python fine_tune.py
```

This resolves to:

```json
{
  "use_pretrain": false,
  "pretrained_path": null,
  "effective_pretrained_path": null
}
```

So the model is still initialized as if `pretrained_path=None`.

## How to enable pretraining

Run:

```bash
python fine_tune.py \
  --use_pretrain true \
  --pretrained_path weights/pre_model.pt \
  --run_name pretrain
```

When enabled, `fine_tune.py` passes:

```python
pretrained_path=args.pretrained_path
```

into `VIT_GIN_Parallel`.

## How to check whether pretraining loaded successfully

Use dry-run mode:

```bash
python fine_tune.py \
  --use_pretrain true \
  --pretrained_path weights/pre_model.pt \
  --run_name check_pretrain \
  --dry_run true
```

Then inspect:

```text
runs/check_pretrain/pretrain_load_audit.json
```

Important fields:

- `checkpoint_exists`
- `encoder_keys_found_count`
- `keys_loaded_count`
- `missing_keys`
- `unexpected_keys`
- `load_success`

If `load_success` is `true`, the checkpoint had matching encoder keys that were loaded through the existing model loading path.

## Files written per run

Each run creates:

```text
runs/{run_name}/config_resolved.json
runs/{run_name}/command.txt
runs/{run_name}/pretrain_load_audit.json
```

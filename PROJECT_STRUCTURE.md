# Project folder convention

This project uses a single-owner rule: each artifact type belongs to one folder only.

- `models/`: model implementation only. Put encoder, GNN, ViT/Former, fusion, edge, loss, decoder, and model registry code here.
- `fieldroaddatapipeline/`: data pipeline code only. Put data reading, preprocessing pipeline logic, DataLoader, samplers, data augmentation, and raw trajectory sample organization code here.
- `scripts/`: runnable entry scripts only. Put experiment launchers, audit entrypoints, export commands, batch automation, and HTML generation entry scripts here.
- `utils/`: reusable helper code only. Put shared Python utilities used by multiple scripts or modules here. Do not put one-off experiment outputs here.
- `runs/`: one training run per subfolder. Each run folder may contain its config, checkpoint, metrics, training curves, and run-local logs needed to reproduce that run.
- `results/`: lightweight cross-run summaries only. Put comparison CSVs, summary tables, leaderboard-style metrics, and stage overview tables here.
- `diagnostics/`: raw diagnostic artifacts only. Put per-point prediction CSVs, edge statistics, threshold sweeps, profiler outputs, and branch/fusion health checks here.
- `analysis/`: interpreted analysis reports only. Put stage READMEs, error attribution reports, audit conclusions, formal-test writeups, and manually reviewed analysis CSVs here.
- `analysis_packs/`: shareable analysis bundles only. Put zip files prepared for ChatGPT or other AI tools here, usually named like `*_for_chatgpt.zip`.
- `outputs/`: final visual outputs only. Put user-openable HTML maps, trajectory prediction pages, real/predicted/error-point maps, and final exported figures here.
- `logs/`: logs only. Put training logs, batch logs, nohup logs, and stage logs here. Avoid separate `batch_logs/` or `nohup_logs/` unless preserving old history.
- `cache/`: reusable computation cache only. Put preprocessed `.pt` datasets, context caches, and feature caches here. This is not a result folder.
- `artifacts/`: frozen reusable artifacts only. Put deployable model bundles, decoder bundles, calibrators, manifests, and label-semantics files here.
- `weights/`: external or fixed weights only. Put pretrained weights, downloaded weights, and fixed checkpoints here. New training-run checkpoints should go under `runs/`.
- `wheat/`: raw or large dataset files only. Put original agricultural trajectory data, split JSONs, adjacency files, and large data assets here.
- `frsmap/`: map rendering library code and assets only. Put HTML templates, static map resources, JavaScript/CSS, and map-rendering support code here.
- `tmp/`: temporary local scratch only. Put disposable intermediate files here; do not use it for formal results.

Local-only folders such as `.idea/`, `__pycache__/`, `.matplotlib_cache/`, `.playwright-cli/`, and notebook checkpoints are environment/cache folders, not project logic.

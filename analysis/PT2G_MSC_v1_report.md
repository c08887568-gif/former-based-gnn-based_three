# PT2G_MSC_v1 Report

## Status

- Experiment: PT2G_MSC_v1_finetune_40ep.
- Module: Multi-scale Segment Context Module after fused image+graph features and before the final head.
- Cache: default cache/wheat_non_iid.
- Graph cache: cache/pretrained_graphs/PT2G_topk3.
- Pretrain mode: edge_weight.

## Metric Comparison

- valid macro-F1 delta: 0.069473.
- valid road-F1 delta: 0.111500.
- test macro-F1 delta: 0.026467.
- test road-F1 delta: 0.043432.
- pred_road_rate: 0.2151193884292321.
- collapse_flag: False.
- final segment_scale: 0.013451552018523216.
- final context_to_fused_ratio: 15.920208477708291.

## Error Analysis

- Error-analysis CSVs were not both available, so detailed error deltas were skipped.

## Decision Notes

- Success mainly depends on valid macro-F1 improvement, valid road-F1 not dropping, stable pred_road_rate, and segment_scale moving away from zero.
- If segment_scale remains near zero, MSC was not effectively used.
- If context_to_fused_ratio is very large and metrics degrade, MSC may be too strong.

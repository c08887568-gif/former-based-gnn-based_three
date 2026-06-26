# PT2/PT3 fine-tune 结构修复报告

## 改了什么

`fine_tune.py` 新增 `--pretrain_mode` 参数，支持 `current`、`edge_weight`、`edge_type_weight`。构建 `VIT_GIN_Parallel` 时会把该参数传入模型，使 PT2 fine-tune 启用 edge_weight 结构，PT3 fine-tune 启用 edge_type_weight 结构。

## 旧问题是什么

旧的 PT2/PT3 fine-tune 虽然加载了对应预训练权重，但 fine-tune 模型仍使用默认 `current` 结构，导致 PT2/PT3 预训练阶段学到的边权/边类型结构没有在监督训练中真正启用。

## 是否重新跑完

PT2/PT3 是否都跑完：是。

## Valid 指标

| group | macro-F1 | road-F1 | field-F1 | accuracy | best_epoch |
|---|---:|---:|---:|---:|---:|
| PT2_edge_weight_pretrain_fixed | 0.820152 | 0.716639 | 0.923664 | 0.879729 | 40 |
| PT3_edge_type_weight_pretrain_fixed | 0.819999 | 0.717162 | 0.922836 | 0.878751 | 37 |

macro-F1 更好：PT2_edge_weight_pretrain_fixed。
road-F1 更好：PT3_edge_type_weight_pretrain_fixed。
field-F1 更好：PT2_edge_weight_pretrain_fixed。

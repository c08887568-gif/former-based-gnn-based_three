# PT1/PT2/PT3 分类头坍塌诊断报告

本报告只读取已有 checkpoint 在 valid 集上做 eval-only 推理，不修改模型结构、不修改训练逻辑、不重新训练。

## 诊断结论

- 严重坍塌：无
- road 偏弱：PT1, PT2, PT3
- macro-F1 正常但 road 预测偏低：PT1
- macro-F1 最好：PT2

## 指标明细

| group | true_road_rate | pred_road_rate | true_field_rate | pred_field_rate | road_f1 | field_f1 | macro_f1 | prediction_entropy | collapse_flag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PT1 | 0.233040 | 0.121406 | 0.766960 | 0.878594 | 0.592530 | 0.912232 | 0.752381 | 0.470517 | road_weak;biased_low_road_prediction |
| PT2 | 0.233040 | 0.191406 | 0.766960 | 0.808594 | 0.716639 | 0.923664 | 0.820152 | 0.459872 | road_weak |
| PT3 | 0.233040 | 0.195646 | 0.766960 | 0.804354 | 0.717162 | 0.922836 | 0.819999 | 0.467813 | road_weak |

## 判定规则

- `pred_road_rate < 0.05` 或 `pred_field_rate < 0.05`：严重坍塌。
- `road_f1 + 0.15 < field_f1`：road 偏弱。
- `field_f1 + 0.15 < road_f1`：field 偏弱。
- `macro_f1 >= 0.70` 但预测 road/field 比例明显低于真实比例：分类偏置。

# PTG test eval-only report

本报告只加载已有 best_model 在 test 集 eval-only 推理，没有重新训练。测试阶段使用对应 PT1G/PT2G/PT3G 的 test 补充边缓存。

| group | test_accuracy | test_macro_f1 | road_f1 | field_f1 | road_recall | field_recall | pred_road_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| PT1G | 0.869559 | 0.781872 | 0.643571 | 0.920172 | 0.542591 | 0.960195 | 0.148929 |
| PT2G | 0.900347 | 0.849641 | 0.762324 | 0.936957 | 0.736339 | 0.945811 | 0.202242 |
| PT3G | 0.893353 | 0.839131 | 0.745737 | 0.932526 | 0.720588 | 0.941243 | 0.202399 |

macro-F1 最好：PT2G。
road-F1 最好：PT2G。
field-F1 最好：PT2G。

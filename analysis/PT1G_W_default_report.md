# PT1G-W + 原始 cache 实验报告

本实验使用 `cache/wheat_non_iid`，加载 PT1 普通预训练权重，使用 PT1 encoder embedding 对原始固定边和 PT1G 补充边计算 cosine similarity，并归一化到 `[0, 1]` 作为静态 `edge_weight`。未使用 PT2 的 EdgeWeightLearner，未修改标签。

| group | cache | valid_macro_f1 | valid_road_f1 | valid_field_f1 | test_macro_f1 | test_road_f1 | test_field_f1 | pred_road_rate | collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PT1G_original | cache/wheat_non_iid | 0.770174 | 0.628818 | 0.911529 | 0.781872 | 0.643571 | 0.920172 | 0.148929 | False |
| PT1G_W_default | cache/wheat_non_iid | 0.756315 | 0.601997 | 0.910633 | 0.715016 | 0.525435 | 0.904596 | 0.117732 | False |

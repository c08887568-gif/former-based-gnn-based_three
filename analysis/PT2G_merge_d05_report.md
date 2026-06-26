# PT2G + normalized merge cache 实验报告

本实验使用已修正归一化的 `cache/wheat_merge_d05_dt10_s1`。merge cache 的模型输入 `points` 已按原始读取管道对每条轨迹执行 MinMax 归一化到 `[0, 1]`，`coordinates` 保留原始经纬度。随后重新进行 PT2 edge_weight 预训练 40 轮，导出 PT2G topk=3 补充边，再 fine-tune 40 轮并运行 test。

| group | cache | valid_macro_f1 | valid_road_f1 | valid_field_f1 | test_macro_f1 | test_road_f1 | test_field_f1 | pred_road_rate | collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PT2G_original | cache/wheat_non_iid | 0.819347 | 0.713875 | 0.924818 | 0.849641 | 0.762324 | 0.936957 | 0.202242 | False |
| PT2G_merge_d05 | cache/wheat_merge_d05_dt10_s1 | 0.737011 | 0.532441 | 0.941581 | 0.744356 | 0.538961 | 0.949751 | 0.078161 | False |

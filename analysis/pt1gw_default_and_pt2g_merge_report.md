# PT1G-W 与 PT2G merge 总结报告

## 指标对比

| group | cache | valid_macro_f1 | valid_road_f1 | valid_field_f1 | test_macro_f1 | test_road_f1 | test_field_f1 | pred_road_rate | collapse |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| PT1G_original | cache/wheat_non_iid | 0.770174 | 0.628818 | 0.911529 | 0.781872 | 0.643571 | 0.920172 | 0.148929 | False |
| PT1G_W_default | cache/wheat_non_iid | 0.756315 | 0.601997 | 0.910633 | 0.715016 | 0.525435 | 0.904596 | 0.117732 | False |
| PT2G_original | cache/wheat_non_iid | 0.819347 | 0.713875 | 0.924818 | 0.849641 | 0.762324 | 0.936957 | 0.202242 | False |
| PT2G_merge_d05 | cache/wheat_merge_d05_dt10_s1 | 0.737011 | 0.532441 | 0.941581 | 0.744356 | 0.538961 | 0.949751 | 0.078161 | False |

## 判断

- PT1G-W 是否提升: 对比 `PT1G_original` 与 `PT1G_W_default`。
- merge cache 是否提升 PT2G: 对比 `PT2G_original` 与 `PT2G_merge_d05`。
- 分类头坍塌判定: `pred_road_rate < 0.05` 或 `pred_field_rate < 0.05`。

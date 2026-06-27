# PT2G_MSC_v1 详细错误分析报告

## 核心指标

- valid macro-F1: 0.8914862499847028
- test macro-F1: 0.8706206603901558
- test road-F1: 0.797176676045406
- test field-F1: 0.9440646447349057
- test road_as_field: 4264
- test field_as_road: 4098
- test high_conf_error: 3607
- long error segments(length>=20): 83
- final segment_scale: 0.013451552018523216
- final context_to_fused_ratio: 15.920208477708291

## 必答问题

1. 当前主要错误类型排序：[('road_as_field', 4264), ('field_as_road', 4098)].
2. road_as_field 是否仍然是主要问题：是，数量 4264。
3. field_as_road 是否同等重要：是，数量 4098。
4. 长段错误还有 83 段；相对 PT2G 的长段变化见 `pt2g_vs_msc_error_delta.csv`。
5. 长段错误归因：inside crop points ratio=0.5702546041529092，cross/near crop points ratio=0.42974539584709076。
6. road_as_field 和弯道/折返关系：road_curve_related segment ratio=0.3945945945945946。
7. field_as_road 和停留密集关系：stationary_dense_related segment ratio=0.14594594594594595。
8. 高置信错点共同特征：见 feature contrast，高置信错点数 3607；top high-conf related features 可从 `feature_error_contrast.csv` 的 high_conf_error_mean 继续查。
9. 错误集中性：top5 trace=0.5783305429323129，top10 trace=0.8668978713226501，top5 grid=0.2833054293231284，top10 grid=0.3879454675914853。
10. MSC 修正点数=3888，新制造错点数=2747。修正/新增错点明细见 msc_fixed_points.csv 和 msc_new_errors.csv。
11. 下一步推荐顺序：PT2G_MSC_v2, PT2G_MSC_HM_v1, PT2G_TSC_v1。

## 和原 PT2G 对比

- accuracy_delta: 0.011965059091242791
- macro_f1_delta: 0.020980037919786643
- road_f1_delta: 0.03485268835063571
- field_f1_delta: 0.007107387488937467
- road_as_field_delta: -1193
- field_as_road_delta: 52
- high_conf_error_delta: 1286
- pred_road_rate_delta: 0.01305565168150502

## 特征差异 Top 10

- all_error: feature_40(-0.273), feature_15(-0.270), feature_22(-0.252), feature_41(0.242), feature_00(-0.224), feature_39(-0.174), feature_17(-0.174), feature_19(-0.157), feature_26(0.155), feature_14(-0.145)
- road_as_field: feature_12(-0.346), feature_37(-0.242), feature_11(-0.217), feature_09(0.202), feature_19(0.170), feature_07(0.166), feature_08(0.163), feature_16(0.163), feature_42(-0.161), feature_18(0.154)
- field_as_road: feature_19(-0.625), feature_39(-0.614), feature_15(-0.596), feature_40(-0.581), feature_14(-0.563), feature_41(0.484), feature_09(-0.468), feature_23(0.463), feature_22(-0.463), feature_03(-0.461)

## 输出文件

- 逐点预测：`diagnostics/PT2G_MSC_SD_v1_error_analysis/valid_point_predictions_with_features.csv` 和 `test_point_predictions_with_features.csv`
- 连续错误段：`diagnostics/PT2G_MSC_SD_v1_error_analysis/error_segments.csv`
- 长段归因：`diagnostics/PT2G_MSC_SD_v1_error_analysis/long_error_segments_with_causes.csv`
- HTML：`outputs/prediction_html/PT2G_MSC_SD_v1/index.html`

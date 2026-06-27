# PT2G_MSC_v1 详细错误分析报告

## 核心指标

- valid macro-F1: 0.8888196998197027
- test macro-F1: 0.8761072669054182
- test road-F1: 0.8057557448254107
- test field-F1: 0.9464587889854258
- test road_as_field: 4094
- test field_as_road: 3911
- test high_conf_error: 2818
- long error segments(length>=20): 86
- final segment_scale: 0.013451552018523216
- final context_to_fused_ratio: 15.920208477708291

## 必答问题

1. 当前主要错误类型排序：[('road_as_field', 4094), ('field_as_road', 3911)].
2. road_as_field 是否仍然是主要问题：是，数量 4094。
3. field_as_road 是否同等重要：是，数量 3911。
4. 长段错误还有 86 段；相对 PT2G 的长段变化见 `pt2g_vs_msc_error_delta.csv`。
5. 长段错误归因：inside crop points ratio=0.5977339236137165，cross/near crop points ratio=0.4022660763862835。
6. road_as_field 和弯道/折返关系：road_curve_related segment ratio=0.37244897959183676。
7. field_as_road 和停留密集关系：stationary_dense_related segment ratio=0.14285714285714285。
8. 高置信错点共同特征：见 feature contrast，高置信错点数 2818；top high-conf related features 可从 `feature_error_contrast.csv` 的 high_conf_error_mean 继续查。
9. 错误集中性：top5 trace=0.620737039350406，top10 trace=0.8871955028107433，top5 grid=0.2485946283572767，top10 grid=0.346158650843223。
10. MSC 修正点数=3888，新制造错点数=2390。修正/新增错点明细见 msc_fixed_points.csv 和 msc_new_errors.csv。
11. 下一步推荐顺序：PT2G_MSC_v2, PT2G_MSC_HM_v1, PT2G_TSC_v1。

## 和原 PT2G 对比

- accuracy_delta: 0.01570872788666222
- macro_f1_delta: 0.02646664443504898
- road_f1_delta: 0.0434317571306404
- field_f1_delta: 0.009501531739457558
- road_as_field_delta: -1363
- field_as_road_delta: -135
- high_conf_error_delta: 497
- pred_road_rate_delta: 0.012877381738866017

## 特征差异 Top 10

- all_error: feature_40(-0.237), feature_15(-0.230), feature_22(-0.202), feature_00(-0.192), feature_41(0.184), feature_17(-0.171), feature_26(0.156), feature_39(-0.142), feature_19(-0.141), feature_06(-0.140)
- road_as_field: feature_12(-0.338), feature_11(-0.231), feature_37(-0.218), feature_09(0.175), feature_22(-0.160), feature_26(0.153), feature_42(-0.145), feature_16(0.140), feature_07(0.140), feature_08(0.137)
- field_as_road: feature_19(-0.602), feature_23(0.541), feature_09(-0.533), feature_08(-0.524), feature_21(0.520), feature_07(-0.510), feature_18(-0.499), feature_39(-0.490), feature_16(-0.489), feature_41(0.437)

## 输出文件

- 逐点预测：`diagnostics/PT2G_MSC_v1_error_analysis/valid_point_predictions_with_features.csv` 和 `test_point_predictions_with_features.csv`
- 连续错误段：`diagnostics/PT2G_MSC_v1_error_analysis/error_segments.csv`
- 长段归因：`diagnostics/PT2G_MSC_v1_error_analysis/long_error_segments_with_causes.csv`
- HTML：`outputs/prediction_html/PT2G_MSC_v1/index.html`

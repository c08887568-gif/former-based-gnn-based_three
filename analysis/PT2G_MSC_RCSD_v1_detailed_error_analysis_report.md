# PT2G_MSC_v1 详细错误分析报告

## 核心指标

- valid macro-F1: 0.8883198324617685
- test macro-F1: 0.8663119050995857
- test road-F1: 0.7924015009380863
- test field-F1: 0.9402223092610851
- test road_as_field: 3803
- test field_as_road: 5049
- test high_conf_error: 3019
- long error segments(length>=20): 98
- final segment_scale: 0.013451552018523216
- final context_to_fused_ratio: 15.920208477708291

## 必答问题

1. 当前主要错误类型排序：[('field_as_road', 5049), ('road_as_field', 3803)].
2. road_as_field 是否仍然是主要问题：否，数量 3803。
3. field_as_road 是否同等重要：是，数量 5049。
4. 长段错误还有 98 段；相对 PT2G 的长段变化见 `pt2g_vs_msc_error_delta.csv`。
5. 长段错误归因：inside crop points ratio=0.6187879201036198，cross/near crop points ratio=0.3812120798963801。
6. road_as_field 和弯道/折返关系：road_curve_related segment ratio=0.3080568720379147。
7. field_as_road 和停留密集关系：stationary_dense_related segment ratio=0.10426540284360189。
8. 高置信错点共同特征：见 feature contrast，高置信错点数 3019；top high-conf related features 可从 `feature_error_contrast.csv` 的 high_conf_error_mean 继续查。
9. 错误集中性：top5 trace=0.661432444645278，top10 trace=0.9036375960234975，top5 grid=0.24118843199276999，top10 grid=0.3351784907365567。
10. MSC 修正点数=3939，新制造错点数=3288。修正/新增错点明细见 msc_fixed_points.csv 和 msc_new_errors.csv。
11. 下一步推荐顺序：PT2G_MSC_v2, PT2G_MSC_HM_v1, PT2G_TSC_v1。

## 和原 PT2G 对比

- accuracy_delta: 0.006826690156353177
- macro_f1_delta: 0.016671282629216466
- road_f1_delta: 0.03007751324331598
- field_f1_delta: 0.0032650520151168427
- road_as_field_delta: -1654
- field_as_road_delta: 1003
- high_conf_error_delta: 698
- pred_road_rate_delta: 0.02786254338775812

## 特征差异 Top 10

- all_error: feature_41(0.210), feature_15(-0.191), feature_40(-0.189), feature_00(-0.148), feature_22(-0.147), feature_06(-0.144), feature_26(0.143), feature_17(-0.134), feature_25(0.119), feature_20(-0.117)
- road_as_field: feature_12(-0.352), feature_22(-0.239), feature_11(-0.231), feature_09(0.222), feature_40(-0.210), feature_37(-0.210), feature_15(-0.201), feature_00(-0.199), feature_08(0.190), feature_07(0.190)
- field_as_road: feature_09(-0.437), feature_23(0.428), feature_08(-0.428), feature_07(-0.420), feature_21(0.418), feature_41(0.410), feature_19(-0.397), feature_18(-0.394), feature_16(-0.391), feature_06(-0.309)

## 输出文件

- 逐点预测：`diagnostics/PT2G_MSC_RCSD_v1_error_analysis/valid_point_predictions_with_features.csv` 和 `test_point_predictions_with_features.csv`
- 连续错误段：`diagnostics/PT2G_MSC_RCSD_v1_error_analysis/error_segments.csv`
- 长段归因：`diagnostics/PT2G_MSC_RCSD_v1_error_analysis/long_error_segments_with_causes.csv`
- HTML：`outputs/prediction_html/PT2G_MSC_RCSD_v1/index.html`

# PT2G_MSC_v1 详细错误分析报告

## 核心指标

- valid macro-F1: 0.8951110784296621
- test macro-F1: 0.8854926037590827
- test road-F1: 0.8205389163516527
- test field-F1: 0.9504462911665128
- test road_as_field: 3766
- test field_as_road: 3640
- test high_conf_error: 2888
- long error segments(length>=20): 79
- final segment_scale: 0.013451552018523216
- final context_to_fused_ratio: 15.920208477708291

## 必答问题

1. 当前主要错误类型排序：[('road_as_field', 3766), ('field_as_road', 3640)].
2. road_as_field 是否仍然是主要问题：是，数量 3766。
3. field_as_road 是否同等重要：是，数量 3640。
4. 长段错误还有 79 段；相对 PT2G 的长段变化见 `pt2g_vs_msc_error_delta.csv`。
5. 长段错误归因：inside crop points ratio=0.6277541863632722，cross/near crop points ratio=0.3722458136367278。
6. road_as_field 和弯道/折返关系：road_curve_related segment ratio=0.33507853403141363。
7. field_as_road 和停留密集关系：stationary_dense_related segment ratio=0.1099476439790576。
8. 高置信错点共同特征：见 feature contrast，高置信错点数 2888；top high-conf related features 可从 `feature_error_contrast.csv` 的 high_conf_error_mean 继续查。
9. 错误集中性：top5 trace=0.6112611396165272，top10 trace=0.8764515257899，top5 grid=0.2624898730758844，top10 grid=0.3679449095328112。
10. MSC 修正点数=4649，新制造错点数=2552。修正/新增错点明细见 msc_fixed_points.csv 和 msc_new_errors.csv。
11. 下一步推荐顺序：PT2G_MSC_v2, PT2G_MSC_HM_v1, PT2G_TSC_v1。

## 和原 PT2G 对比

- accuracy_delta: 0.021990121747884395
- macro_f1_delta: 0.035851981288713475
- road_f1_delta: 0.05821492865688238
- field_f1_delta: 0.013489033920544569
- road_as_field_delta: -1691
- field_as_road_delta: -406
- high_conf_error_delta: 567
- pred_road_rate_delta: 0.013475110370067434

## 特征差异 Top 10

- all_error: feature_40(-0.248), feature_15(-0.238), feature_22(-0.230), feature_41(0.221), feature_00(-0.216), feature_17(-0.167), feature_26(0.165), feature_39(-0.157), feature_12(-0.152), feature_14(-0.134)
- road_as_field: feature_12(-0.333), feature_11(-0.233), feature_37(-0.212), feature_22(-0.185), feature_40(-0.179), feature_00(-0.177), feature_26(0.162), feature_15(-0.157), feature_09(0.136), feature_24(-0.128)
- field_as_road: feature_41(0.464), feature_19(-0.363), feature_15(-0.359), feature_39(-0.353), feature_40(-0.349), feature_14(-0.315), feature_22(-0.296), feature_00(-0.274), feature_23(0.264), feature_03(-0.261)

## 输出文件

- 逐点预测：`diagnostics/PT2G_MSC_RC_v1_error_analysis/valid_point_predictions_with_features.csv` 和 `test_point_predictions_with_features.csv`
- 连续错误段：`diagnostics/PT2G_MSC_RC_v1_error_analysis/error_segments.csv`
- 长段归因：`diagnostics/PT2G_MSC_RC_v1_error_analysis/long_error_segments_with_causes.csv`
- HTML：`outputs/prediction_html/PT2G_MSC_RC_v1/index.html`

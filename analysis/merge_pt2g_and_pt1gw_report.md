# merge PT2G and PT1G-W 汇总报告

## 当前结论

本次已完成代码实现、43 维复现审计、merge cache 构建和 edge_weight 通路验证；两个 40 轮正式实验尚未在本地运行，因此新实验指标暂时为空，不能下训练结论。

## 原始 PT2G 指标

- valid accuracy: 0.8809241349661838
- valid macro-F1: 0.8193466190207233
- valid road-F1: 0.7138753906461694
- valid field-F1: 0.9248178473952771
- test accuracy: 0.9003471020647854
- test macro-F1: 0.8496406224703692
- test road-F1: 0.7623239876947703
- test field-F1: 0.9369572572459682

## PT2G + merge cache

- 状态: pending_training
- cache: `cache/wheat_merge_d05_dt10_s1`
- 目的: 比较连续近邻重复点合并是否提升 PT2G。

## 原始 PT1G 指标

- valid accuracy: 0.8571149700011451
- valid macro-F1: 0.7701736422338402
- valid road-F1: 0.6288180610889774
- valid field-F1: 0.911529223378703
- test accuracy: 0.8695588343243045
- test macro-F1: 0.7818718478772397
- test road-F1: 0.6435714490386544
- test field-F1: 0.920172246715825

## PT1G-W + default cache

- 状态: pending_training
- cache: `cache/wheat_non_iid`
- 目的: 比较给 PT1G 固定边赋 PT1 encoder cosine 权重是否提升。

## 坍塌检查

当前新实验未训练，无法判断是否坍塌。原始 PT1G / PT2G test 预测道路比例分别为 0.14892880737408373 和 0.20224200669036607，均未触发 `< 0.05` 的严重坍塌规则。

## 是否建议保留

- merge 预处理: 需要等待 PT2G_merge_d05 test 指标，尤其看 macro-F1、road-F1 是否超过原始 PT2G。
- PT1G-W: 需要等待 PT1G_W_default test 指标，尤其看 road-F1 和 pred_road_rate 是否改善。

汇总表路径: `results/merge_pt2g_and_pt1gw_summary.csv`。

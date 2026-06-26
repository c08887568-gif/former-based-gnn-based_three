# wheat merge d05 feature audit report

## 审计对象

- cache: `cache/wheat_merge_d05_dt10_s1/`
- 对比来源: `wheat/sampled_wheat_43/*.xlsx`
- 审计输出: `diagnostics/wheat_merge_d05_feature_audit.csv`

## 审计规则

- 除第 6 维外，检查 cache 中 42 个特征维度是否等于对应 `sampled_wheat_43` merge group 的组内均值。
- 第 6 维，即列索引 `5`，重新按合并后经纬度序列计算 Haversine 距离和，并与 cache 内值比较。
- 标签检查为 merge group 内多数标签。
- coordinates 检查为 `points[:, 41:43]`。
- merge_mapping 检查索引是否有效。

## 总结

```json
{
  "traces": 100,
  "passed": 100,
  "failed": 0,
  "points": 587686,
  "feature_mean_dims_max_abs_diff": 0.0,
  "feature_mean_dims_mean_abs_diff": 0.0,
  "feature6_max_abs_diff": 0.0,
  "feature6_mean_abs_diff": 0.0,
  "label_mismatch_count": 0,
  "merge_mapping_invalid_count": 0,
  "nan_count": 0,
  "coordinate_max_abs_diff": 0.0
}
```

## 最大非第 6 维误差 Top 5

```csv
split,trace_id,points,feature_mean_dims_max_abs_diff,feature_mean_dims_mean_abs_diff,pass_flag
train,wheat/sampled_wheat_43/wheat_1_harvestor_99.xlsx,5479,0,0,True
train,wheat/sampled_wheat_43/wheat_1_harvestor_110.xlsx,5008,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_63.xlsx,7668,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_77.xlsx,3384,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_92.xlsx,3653,0,0,True
```

## 第 6 维误差 Top 5

```csv
split,trace_id,points,feature6_max_abs_diff,feature6_mean_abs_diff,pass_flag
train,wheat/sampled_wheat_43/wheat_1_harvestor_99.xlsx,5479,0,0,True
train,wheat/sampled_wheat_43/wheat_1_harvestor_110.xlsx,5008,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_63.xlsx,7668,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_77.xlsx,3384,0,0,True
valid,wheat/sampled_wheat_43/wheat_1_harvestor_92.xlsx,3653,0,0,True
```

## 未通过轨迹

```csv
none
```

## 结论

全部轨迹通过审计，误差仅为 float32 量级。

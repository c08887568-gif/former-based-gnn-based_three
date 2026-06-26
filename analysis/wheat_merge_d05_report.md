# wheat merge duplicate cache report

本次只构建连续近邻重复点合并后的 torch cache，不训练模型，不覆盖 `wheat/sampled_wheat`，不覆盖 `wheat/sampled_wheat_43`，不生成新的 Excel 中转目录。

## 输出

- cache: `cache/wheat_merge_d05_dt10_s1/`
- train/valid/test cache item 数: `{'train': 455, 'valid': 105, 'test': 77}`
- audit: `diagnostics/wheat_merge_d05_audit.csv`

## 总体点数变化

- 总原始点数: `769858`
- 总合并后点数: `587686`
- 总减少比例: `0.236631`

## split 点数变化

- train: original=541481, merged=421185, reduction_rate=0.222161, cache_items=455
- valid: original=133016, merged=97554, reduction_rate=0.266600, cache_items=105
- test: original=95361, merged=68947, reduction_rate=0.276990, cache_items=77

## mixed-label merge groups

- mixed-label merge groups: `175`
- merge groups: `53504`
- mixed-label rate: `0.003271`
- 判断: `不高，仅作为诊断记录`

## edge 变化

- original_edges: `4796858`
- final_edges: `3531754`
- self_loops_removed: `687178`
- duplicate_edges_removed: `577926`
- temporal_edges_added: `0`
- 判断: `edge 数存在明显下降，主要来自合并后自环删除和去重；已补充时间连续双向边`

## 第 6 维

- 第 6 维，即代码列索引 `5`，已按合并后的新经纬度序列重新计算: `True`。
- 公式: `feature_6(i) = sum(distance(point_i, point_(i+k)))`, `k=1..10`；如果未来不足 10 个点，则改用前面最多 10 个点。
- 距离使用 Haversine 球面距离，单位米。
- 其它 42 维来自对应 `sampled_wheat_43` 原始行的组内均值。
- 生成模型输入 `points` 前，已按原始读取管道的方式对每条合并后轨迹的 43 个输入列执行 MinMax 归一化到 `[0, 1]`。
- `coordinates` 仍保存未归一化的原始经纬度，供 HTML 和诊断使用。

## 空轨迹或异常样本

- 异常轨迹数量: `0`
- 异常轨迹: `[]`

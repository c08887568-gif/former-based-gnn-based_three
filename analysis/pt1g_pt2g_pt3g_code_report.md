# PT1G / PT2G / PT3G 代码实现报告

本次只实现代码，不正式训练；baseline 默认行为保持不变，不传 `--graph_cache_path` 时，`fine_tune.py` 仍然只使用原始数据中的 `edge_index`。

## 1. 三组新增实验含义

- PT1G：普通 masked reconstruction 预训练 + 预训练 encoder 发现的补充边。
- PT2G：边权预训练 + 预训练 encoder 发现的补充边。
- PT3G：边类型 + 边权预训练 + 预训练 encoder 发现的补充边。

三组实验的 fine-tune 阶段都使用：原始数据边 + 预训练发现的补充边。

## 2. 补充边如何生成

新增 `scripts/export_pretrained_graph_edges.py`，用于加载指定预训练权重，并对 `train` / `valid` 每条轨迹导出补充边缓存。

生成规则如下：

1. 从缓存数据读取轨迹点特征和原始 `edge_index`。
2. 使用加载了预训练权重的 `VIT_GIN_Parallel` encoder 提取每个点的 graph embedding。
3. 计算点与点之间的 embedding 余弦相似度。
4. 排除自环和原始图中已经存在的边。
5. 对每个点选择相似度最高的 `topk` 个点作为补充边，默认 `topk=3`。
6. 只保留相似度为正的补充边。
7. 全程不使用真实标签。

导出的每条轨迹至少包含：

- `trace_id`
- `extra_edge_index`
- `extra_edge_weight`
- `extra_edge_type_prob`，仅在 edge type learner 可用时保存
- `original_num_edges`
- `extra_num_edges`

额外保存了 `sample_index` 和 `points_hash`，用于同一个 `trace_id` 被切成多个样本时进行更稳的匹配。

## 3. fine_tune 如何读取增强图

`fine_tune.py` 新增参数：

```bash
--graph_cache_path
```

默认值为 `None`。当该参数为空时，训练和验证仍然只使用原始边。

当指定 `--graph_cache_path cache/pretrained_graphs/PT1G_topk3` 这类目录时，`fine_tune.py` 会读取：

- `cache/pretrained_graphs/PT1G_topk3/train.pt`
- `cache/pretrained_graphs/PT1G_topk3/valid.pt`

训练时按 `trace_id` 匹配对应轨迹，将：

```text
edge_index = original_edge_index + extra_edge_index
```

然后继续走原来的训练流程。

合并后会按有向边 `(src, dst)` 去重；如果补充边和原始边重复，优先保留原始边。每次合并的去重统计会写入 `diagnostics/graph_cache_dedup_audit.csv`，包含 `original_edges`、`extra_edges_before_dedup`、`duplicate_edges_removed`、`merged_edges_after_dedup`。

## 4. edge_weight 当前说明

当前版本会在导出缓存中保存 `extra_edge_weight`，它来自 embedding 相似度；PT3G 在可用时还会保存 `extra_edge_type_prob`。

但是 fine-tune 的 message passing 目前只合并并使用 `extra_edge_index`，没有把 `extra_edge_weight` 显式传入 GNN 传播。因此本版本验证的是“预训练发现的补充边是否有效”，不是验证“补充边权重是否有效”。

## 5. 本次范围

本次没有：

- 修改标签；
- 加入 T2C；
- 加入段级模块；
- 修改 loss；
- 修改默认 baseline 行为；
- 正式启动 PT1G / PT2G / PT3G 训练。

新增命令生成脚本 `scripts/run_pretrained_graph_ablation_pt1g_pt3g.py` 默认只打印命令，不会自动训练。

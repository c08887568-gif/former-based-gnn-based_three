# PT0-PT3 预训练代码检查与设计说明

## 当前代码事实

1. 当前 `pretrain.py` 使用 `models.Pretrain.Pretrain_Parallel` 做 masked reconstruction 预训练。输入轨迹点先进入 image/VIT 分支和 graph/GIN 分支，然后 decoder 分别重建被 mask 的 image patch 与 graph 节点特征，最终 loss 是 image reconstruction loss 与 graph reconstruction loss 的平均。

2. 当前 `Pretrain_Parallel` 使用 `VIT_GIN_Parallel` 作为 encoder。`VIT_GIN_Parallel` 内部包含 `vision = VIT(...)` 和 `graph = GIN(...)` 两个分支。

3. 当前 graph 分支存在 `GIN.graph_random_masking()`。它会随机保留一部分节点，基于保留节点重建 masked 后的 `edge_index`，并返回 `mask` 与 `ids_restore` 供 decoder 还原顺序。

4. 当前 `models/mysg_conv3.py` 的 `SGConv.message()` 支持 `edge_weight.view(-1, 1) * x_j`，也就是消息传播时可以按边权缩放邻居节点特征。

5. 当前 `SGConv(is_attn=True)` 会计算 attention，并基于 attention 结果形成 masked attention adjacency；但 attention 数值没有作为一套可审计、可保存、可迁移的 `edge_weight` 统计产物进入预训练权重分析流程。

## 四组实验定位

PT0_no_pretrain 是监督训练 baseline 对照组，不加载任何预训练权重，用当前 `fine_tune.py` 和当前 baseline 模型直接训练。

PT1_current_masked_pretrain 使用当前已有的 image + graph masked reconstruction 预训练，不额外学习可解释边权或边类型。

PT2_edge_weight_pretrain 在预训练阶段加入自监督边权学习。每条边的特征由 `x_i`、`x_j`、`abs(x_i - x_j)` 拼接得到，再由 MLP 输出 `sigmoid` 边权，边权只通过 masked reconstruction loss 学习，不使用 road/field 标签。

PT3_edge_type_weight_pretrain 在 PT2 的基础上进一步学习软边类型。每条边由 `MLP_type` 输出 K=4 个软类型概率，再结合可学习 `type_weight` 和边自身 gate 形成最终传播边权，同样只使用 masked reconstruction loss，不使用真实标签或人工 edge type 标签。

## PT2/PT3 与 PT1 的区别

PT1 的 graph masked reconstruction 主要复用当前 GIN/SGConv 传播逻辑。PT2/PT3 仍然是同一个 masked reconstruction 任务，但让图传播阶段的边权由模型根据节点关系自监督学习。

PT2 只学习“这条边传播多少信息”。PT3 同时学习“这条边更像哪一种传播关系”和“这条边传播多少信息”，并记录类型坍缩风险。

## 本阶段边界

本阶段只实现 PT0-PT3 所需代码、脚本、审计和结果模板，不正式训练；不加入 T2C、段级模块，也不引入旧版 edge/fusion/loss 实验框架。

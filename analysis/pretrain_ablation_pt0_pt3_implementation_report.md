# PT0-PT3 预训练消融实现报告

## 四组实验

PT0_no_pretrain：不使用预训练，直接用当前 baseline 做监督训练，是后续比较预训练收益的对照组。

PT1_current_masked_pretrain：使用当前已有 image + graph masked reconstruction 预训练。image 分支重建 masked patch，graph 分支重建 masked 节点特征。

PT2_edge_weight_pretrain：在 masked reconstruction 预训练中加入自监督边权学习。边权由 `concat(x_i, x_j, abs(x_i - x_j))` 经 MLP 和 sigmoid 得到，再进入 SGConv message passing。

PT3_edge_type_weight_pretrain：在 PT2 基础上加入软边类型学习。每条边学习 K=4 个类型概率，并结合可学习 type weight 与边 gate 得到最终传播边权。

## 为什么 PT0 是对照组

PT0 不加载任何预训练参数，直接监督训练当前 baseline。因此 PT1/PT2/PT3 的 fine-tune 指标都应该和 PT0 比较，才能判断预训练是否真正带来收益。

## PT1 当前预训练学的是什么

PT1 学的是轨迹点特征在局部序列和图结构上的可恢复表示。它不使用 road/field 标签，只通过 image reconstruction 与 graph reconstruction 约束 encoder。

## PT2 为什么是自监督边权学习

PT2 的边权不是来自真实标签，也不是人工规则标签，而是由节点两端特征关系自动生成，并且只通过 masked reconstruction loss 反向传播更新。因此它属于自监督边权学习。

## PT3 为什么是自监督边类型 + 边权学习

PT3 的 edge type 是 softmax 输出的软类型概率，没有任何人工 edge type supervision。类型概率、type weight 和 edge gate 都只由 masked reconstruction 训练，所以它是自监督边类型与边权联合学习。

## 标签使用边界

PT2/PT3 仍然只使用 masked reconstruction，不使用 road/field 标签；标签定义保持 `0 = road`、`1 = field`，但不会进入预训练 loss。

## 本次实现范围

本次只实现代码、脚本、dry-run、统计文件模板和命令生成，没有正式训练。

预训练脚本默认 `pretrain_epochs = 40`。当前 `fine_tune.py` 暂时没有 `--epochs` 参数，因此 PT0-PT3 的 fine-tune 示例命令沿用 `fine_tune.py` 当前训练轮数设置，不在本阶段强行改监督训练逻辑。

下一步应把命令复制到云 GPU 上运行，先分别执行 PT1/PT2/PT3 的 dry-run，确认 cache、模型 forward 和 loss 正常，再启动正式预训练与 fine-tune。

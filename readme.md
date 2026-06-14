以下是文件结构介绍：
python代码文件简介：
1.dataset.py包含了数据结构定义。
2.draw_map.py包含了绘制遥感影像地图的运行代码。(必须要有edge浏览器，如果生成图是空白的，去mapbox注册一个api替换掉url)
3.pretrain.py包含预训练代码。
4.fine_tune.py包含了训练模型的代码。
5.predict.py包含了预测结果的代码。
文件夹简介：
1.fieldroaddatapipeline中包含了所有数据处理过程,用法见DataRetrievalDemonstration.py。
2.frsmap中包含了遥感影像可视化代码。
3.models中包含了模型文件。
4.utils中包含了训练过程中用到的方法。关键用法：运行run_model.py即可完成训练，但这只是个demo，具体超参数需要自己调整。
5.logs中包含了训练日志。
6.weights中包含了训练好的参数。
7.outputs中包含了所有可视化以及有用的输出。

这是transformer和gnn的混合架构模型
还有协同loss和权重正交loss

看代码学习一下，我是如何：
【1】将不同网络组成混合架构
【2】我是如何预训练（这个模型预训练创新最高）
【3】我是如何改loss

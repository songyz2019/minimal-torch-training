# minimal-torch-training
A Minimal PyTorch training and testing framework for beginners.  
适用于初学者的最简单PyTorch模型训练测试代码框架。


## 代码解读
一个基本的(用于分类的)深度学习代码应分为两个过程: 训练(`train.py`)和测试(`test.py`)

每个过程由下面几个变量组成:
1. `dataset`: 数据的来源,需要关心的是它的`train`参数,如果为`True`,表示读取训练集,否则为测试集。示例代码中用的是torchvision内置的MINIST数据集,如果要用非内置数据集,可以重载`torch.utils.Dataset`类实现  
2. `dataloader`: 数据的读取方式,对于训练过程,我们需要将数据顺序打乱(`shuffle=True`), 并一组一组的读取数据(`batch_size=64`),测试过程则不需要
3. `model`: 模型
4. `loss`: 损失
5. `optimizer`: 优化器,它根据`loss`优化`model`

<!--
## Basic Deeplearning Engineering for Classification
A simple deeplearning project should contain:
1. `dataloader`: where our data come from.
2. `model`: basically a map from x to y, where x is our data, for example, an image; and y is our target or result
3. `optimizer`: 
-->
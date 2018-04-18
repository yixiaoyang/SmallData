### numpy.random.randn
返回随机的正态分布样本，参数为矩阵的行和列

### zip
zip函数接受任意多个（包括0个和1个）序列作为参数，返回一个tuple列表

```python
>>> x = [1, 2, 3]
>>> y = [4, 5, 6]
>>> zipped = zip(x, y)
>>> zipped
[(1, 4), (2, 5), (3, 6)]
>>> x2, y2 = zip(*zipped)
>>> x == list(x2) and y == list(y2)
True
```

### list切片
```python
>>> b=[[1,2,3],[4,5,6],[7,8,9]]
>>> print(b[:-1])
[[1, 2, 3], [4, 5, 6]]
```


### 随机梯度下降算法
梯度下降（GD）是最小化风险函数、损失函数的一种常用方法，随机梯度下降（Stochastic Gradient Descent, SGD）和批量梯度下降（Batch Gradient Descent, BGD）是两种迭代求解思路

#### 批量梯度下降
最小化所有训练样本的损失函数，使得最终求解的是全局的最优解，即求解的参数是使得风险函数最小。

#### 随机梯度下降
最小化每条样本的损失函数，虽然不是每次迭代得到的损失函数都向着全局最优方向， 但是大的整体的方向是向全局最优解的，最终的结果往往是在全局最优解附近。

### tf.nn.conv2d

这个函数的功能是：给定4维的input和filter，计算出一个2维的卷积结果。函数的定义为
```
def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
        data_format=None, name=None)
```
前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu

`input`:待卷积的数据。格式要求为一个张量，[batch, in_height, in_width, in_channels]. 分别表示 批次数，图像高度，宽度，输入通道数。

`filter`:卷积核。格式要求为[filter_height, filter_width, in_channels, out_channels]. 分别表示 卷积核的高度，宽度，输入通道数，输出通道数。

`strides`:一个长为4的list. 表示每次卷积以后卷积窗口在input中滑动的距离

`padding`:有SAME和VALID两种选项，表示是否要保留图像边上那一圈不完全卷积的部分。如果是SAME，则保留

`use_cudnn_on_gpu`:是否使用cudnn加速。默认是True

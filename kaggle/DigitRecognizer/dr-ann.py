#!/usr/bin/python
# -*- coding: utf-8 *-*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

'''
bp网路解决数字识别问题

三层bp网络

输入向量 -> 输入层 -> 隐藏层 -> 输出层 -> 输出向量

000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783
'''

# 28*28 = 784
IMG_WEIGHT = 28
IMG_COUNT = 0
IMG_HIGHT = 28

def plot_image(img):
    image = img.reshape(IMG_WEIGHT,IMG_HIGHT)
    plt.axis('off')
    plt.imshow(image,cmap=cm.binary)
    plt.show()
    print ("plot done")

# 利用panda工具读取数据
data = pd.read_csv('./data/train.csv')
print (data.head())

images = data.iloc[:,1:].values
images = images.astype(np.float)
shape = images.shape
print (type(images))
print (type(images[0][1]))
print (shape[0],shape[1])

# 归一化，将所有值乘以1/255
np.multiply(images,1.0/255.0)

# 显示图片
plot_image(images[3])

# 提取labels
labels = np.unique(data.iloc[:,0:1].values)
print (labels)

# 卷积神经处理

# 三层BP网络
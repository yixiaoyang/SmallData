#!/usr/bin/python
# -*- coding: utf-8 *-*

'''
使用多层神经网络进行分类
'''

import os
import math
import time
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial

# 全局变量
g_train_set = None
g_train_target = None
g_test_set = None
g_centroids = None
g_chars = [0,2]
g_data_step = [0, 50, 50, 100]
#g_chars = [1,3]
#g_data_step = [50, 100, 100, 150]

global g_ppn

class Config:
    DATA_FILE = "/devel/git/github/SmallData/LearnMLWithPython/ch02/iris/uic-iris-data/iris.data"
    CLASSES = {
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }
    LABEL_NAMES = ["SepalLen", "SepalWidth", "PetalLen", "PetalWidth", "Classes"]


class Perceptron:
    '''
    基本感知器

    Attributes
    -----------
    '''

    def __init__(self, speed=0.1, train_limit=30, debug=False):
        self.errors = []
        self.speed = speed
        self.debug = debug
        self.train_limit = train_limit
        self.w0 = []

    # 迭代训练函数
    # x:输入向量
    # y:目标向量
    def train(self, x, y, init_param=True):
        if init_param:
            # 每个列数即一个属性，每个属性作为一层，分配一个权值，最后加上一个bias偏置的权值
            # bias可以看作一组（[1,1,..,1],b）的输入，数学上等价
            self.weight = np.zeros(x.shape[1])
            self.bias = 0

        for train_idx in range(self.train_limit):
            errs = 0
            for i, val in enumerate(zip(x, y)):
                xi, ti = val[0], val[1]
                # 计算步进向量
                update = self.speed * (ti - self.active(xi))
                # 更新权重向量和偏置
                self.weight += update * xi
                # bias可以看作一组（[1,1,..,1],b）的输入，数学上等价，所以也需要加上步进
                self.bias += update
                errs += int(update != 0.0)
            self.errors.append(errs)
            self.w0.append(self.weight[0])
            print(self.weight, self.bias, errs)

    # 输入计算函数，返回所有信号输出组成的向量
    def input(self, x):
        # 向量点积计算输入向量xi的输出
        return np.dot(x, self.weight) + self.bias

    # 激活函数，简单的阈值函数即可进行分类
    def active(self, x):
        return np.where(self.input(x) >= 0.0, 1, -1)

    def statics(self):
        # 统计迭代过程错误次数（理想情况下错误为0即达到收敛目的）
        plot.plot(range(1, len(self.errors) + 1), self.errors, marker='o', color="c")
        plot.plot(range(1, len(self.w0) + 1), self.w0, marker="^", color="k")
        plot.xlabel('Iterations')
        plot.ylabel('W0/Missclassifications')
        plot.show()


def data_import(file, delimiter):
    return np.genfromtxt(file, delimiter=delimiter)

def ann_test(x,y,ppn):
    for idx,xi in enumerate(x):
        output = np.dot(xi, ppn.weight) + ppn.bias
        print(idx, xi, y[idx][0], output[0])
    pass

# 使用神经网络进行训练，迭代次数限制
def ann_train():
    global g_train_set
    global g_centroids
    global g_ppn

    x = g_train_set[:, g_chars]
    y = g_train_set[:, [4]]
    g_ppn = Perceptron(speed=0.1, train_limit=50)
    g_ppn.train(x, y)
    ann_plot(g_ppn)
    ann_test(x,y,g_ppn)

def ann_init():
    global g_train_set

    datafile = Config.DATA_FILE
    data = data_import(datafile, ',')

    # 读取并合并训练集数组
    g_train_set = np.hstack((data[g_data_step[0] :g_data_step[1],  0:4], np.full((50, 1), -1.0)))
    g_train_set = np.vstack((g_train_set, np.hstack((data[g_data_step[2]:g_data_step[3], 0:4], np.full((50, 1), 1.0)))))
    # g_train_set = np.hstack((data[50:100,  0:4], np.full((50, 1), -1.0)))
    # g_train_set = np.vstack((g_train_set, np.hstack((data[100:150, 0:4], np.full((50, 1), 1.0)))))

def ann_plot(ppn):
    global g_train_set

    my_set = g_train_set

    plot.scatter(my_set[0  :50 ,g_chars[0]],my_set[0  :50 ,g_chars[1]],marker="^",color="k")
    plot.scatter(my_set[50 :100,g_chars[0]],my_set[50 :100,g_chars[1]],marker="o",color="m")
    plot.xlabel(Config.LABEL_NAMES[g_chars[0]])
    plot.ylabel(Config.LABEL_NAMES[g_chars[1]])

    print(ppn.weight[0],ppn.weight[1],ppn.bias[0])
    fit = np.array([(-ppn.bias[0])/ppn.weight[1], -ppn.weight[0]/ppn.weight[1]])
    fnd = np.poly1d(fit)
    fx = np.linspace(0,8)
    plot.plot(fx,fnd(fx),linewidth=1)

    plot.autoscale(tight=True)
    plot.grid()
    plot.show()

if __name__ == "__main__":
    # 数据导入
    ann_init()
    # 训练数据
    ann_train()

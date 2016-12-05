#!/usr/bin/python
# -*- coding: utf-8 *-*

'''
 Attribute Information:
   0. sepal length in cm
   1. sepal width in cm
   2. petal length in cm
   3. petal width in cm
   4. class:
      -- 0: Iris Setosa
      -- 1: Iris Versicolour
      -- 2: Iris Virginica
'''

import os
import sys
import math
import time
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial
from matplotlib.animation import FuncAnimation

# 全局变量
g_train_set = None
g_train_target = None
g_test_set = None
g_chars = [0,2]
g_data_step = [0, 50, 50, 100]
#g_chars = [1,3]
#g_data_step = [50, 100, 100, 150]
g_ppn = None

# 一个数据背景，一个数据分割
plot_fig, plot_ax = plot.subplots()
plot_line, = plot_ax.plot(0, 0, 'r-', linewidth=2)

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

    def __init__(self, speed=1, train_limit=30, debug=False):
        self.errors = []
        self.speed = speed
        self.debug = debug
        self.train_limit = train_limit
        self.w0 = []
        self.train_count = 0

    # 迭代训练函数
    # x:输入向量
    # y:目标向量
    def train(self, x, y):
        if self.train_count==0:
            # 每个列数即一个属性，每个属性作为一层，分配一个权值，最后加上一个bias偏置的权值
            # bias可以看作一组（[1,1,..,1],b）的输入，数学上等价
            self.weight = np.zeros(x.shape[1])
            self.bias = 0

        self.train_count += 1
        if self.train_count < self.train_limit:
            errs = 0
            for i, val in enumerate(zip(x, y)):
                xi, ti = val[0], val[1]
                # 计算步进向量
                diff = ti - self.active(xi)
                update = self.speed * diff
                # 更新权重向量和偏置
                self.weight += update * xi
                # bias可以看作一组（[1,1,..,1],b）的输入，数学上等价，所以也需要加上步进
                self.bias += update
                errs += int(update != 0.0)
                #print diff
            self.errors.append(errs)
            self.w0.append(self.weight[0])
            print(self.train_count, self.weight, self.bias, errs)

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

def ann_init():
    global g_train_set
    global g_ppn

    datafile = Config.DATA_FILE
    data = data_import(datafile, ',')

    # 读取并合并训练集数组
    g_train_set = np.hstack((data[g_data_step[0] :g_data_step[1],  0:4], np.full((50, 1), -1.0)))
    g_train_set = np.vstack((g_train_set, np.hstack((data[g_data_step[2]:g_data_step[3], 0:4], np.full((50, 1), 1.0)))))
    # g_train_set = np.hstack((data[50:100,  0:4], np.full((50, 1), -1.0)))
    # g_train_set = np.vstack((g_train_set, np.hstack((data[100:150, 0:4], np.full((50, 1), 1.0)))))

    # 准备数据
    g_ppn = Perceptron(speed=0.1, train_limit=50)

def ann_plot_data():
    global g_train_set
    global plot_fig
    global plot_ax
    global g_chars

    my_set = g_train_set
    plot_ax.scatter(my_set[0  :50 ,g_chars[0]],my_set[0  :50 ,g_chars[1]],marker="^",color="k")
    plot_ax.scatter(my_set[50 :100,g_chars[0]],my_set[50 :100,g_chars[1]],marker="o",color="m")
    plot_ax.set_xlabel(Config.LABEL_NAMES[g_chars[0]])
    plot_ax.set_ylabel(Config.LABEL_NAMES[g_chars[1]])

def plot_update(fit):
    global plot_line
    global g_train_set
    global g_ppn
    global g_chars

    # 训练数据
    g_ppn.train(g_train_set[:, g_chars], g_train_set[:, [4]])

    # 更新分割线
    fit = np.array([-g_ppn.weight[0]/g_ppn.weight[1],(-g_ppn.bias[0])/g_ppn.weight[1]])
    fnd = np.poly1d(fit)
    fx = np.linspace(0,8)
    plot_line.set_xdata(fx)
    plot_line.set_ydata(fnd(fx))

    str = ", y=%0.2f*x0+%0.2f*x1+%0.2f=0, train %d"%(g_ppn.weight[0],g_ppn.weight[1],g_ppn.bias[0],g_ppn.train_count)
    plot_ax.set_xlabel(Config.LABEL_NAMES[g_chars[1]]+str)

if __name__ == "__main__":
    # 导入训练集合
    ann_init()
    ann_plot_data()

    # FuncAnimation 会在每一帧都调用“update” 函数,  在这里设置一个10帧的动画，每帧之间间隔500毫秒
    anim = FuncAnimation(plot_fig, plot_update, frames=np.arange(0, 8), interval=500)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        # 会一直循环播放动画
        plot.show()

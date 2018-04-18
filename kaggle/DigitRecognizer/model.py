#!/usr/bin/python
# -*- coding: utf-8 *-*

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# BP网络模型
class ANN:
    """
    sizes   描述神经网络的层数和每层的神经元个数，如[1,2,3]描述的是一个三层网络，三层中的
            神经元个数分别为1,2,3。第一个是输入层的数目，从第二层开始才有权重
    biases  使用随机数初始化bias参数
    """
    def __init__(self, sizes):
        self.layers = len(sizes)
        slef.sizes = sizes
        # 初始化权重和偏置矩阵
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(x,y)
            for x,y in zip(sizes[1:],sizes[:-1]])]
        # 损失
        self.loss = 0
        # 学习速率
        self.speed = 0.01

    """
    正向传播
    input   输入矩阵
    return  返回网络输出
    """
    def forward(slef,i):
        return active(np.dot(w,i)+b) for w,b in zip(self.weights, self.biases)

    """
    批量梯度下降 Stochastic gradient descent
    tdata   训练数据, training data
    epochs  迭代次数
    speed   下降速度
    ddata   结果数据，desired data

    """
    def gradient_descent(self,tdata, epochs, batch_size, speed, ddata):


    """
    反向传播 back propagation algorithm
    """
    def backporp(slef,input):
        pass

    def bp_train(self):
        # 正向传播

        # 反向传播

        # 误差分析

        # 退出
        pass


    # sigmoid作为激活函数
    def active(o):
        return 1.0/(1.0+np.exp(-o))

    # 预测输入x的输出
    def predict(x):
        hIn = x
        hOut = None
        hActive = None
        # 正向传播: 隐藏层的输入为输入层向量，输出层的输入为隐藏层的输出
        for h in range(self.hCount+1):
            # 隐藏层h的输出计算
            self.reo[h] = hIn.dot(self.w[h])+self.b[h]
            hActive = active(hOut)
            # h层的输出作为下一层的输入
            hIn = hActive
        exp_scores = np.exp(hOut)
        # np.sum axis=1按行求和
        probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)

        # 计算损失

    # 损失计算函数
    def calc_loss(self):
        pass

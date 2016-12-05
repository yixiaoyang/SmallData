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
import math
import time
import numpy as np
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

from scipy import spatial

# 全局变量
g_train_set = None
g_test_set  = None
g_centroids = None

class Config:
    DATA_FILE="./iris.data"
    CLASSES = {
        "Iris-setosa":0,
        "Iris-versicolor":1,
        "Iris-virginica":2
    }
    LABEL_NAMES = ["SepalLen","SepalWidth","PetalLen","PetalWidth","Classes"]

def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

# euclidean distance of two 1-D vectors
def vec_distance(a,b):
    return spatial.distance.euclidean(a, b)
def test_vec_distance():
    a=np.array([[1,2,3,4]])
    b=np.array([[1,3,5,4]])
    # = sqrt(0+1+4+0)=sqrt(6)=2.2360679775
    print(vec_distance(a,b))

def kmeans_test():
    global g_test_set
    global g_centroids

    distances = np.zeros((3,75))
    success,mistake=np.zeros(3),np.zeros(3)
    target_classes, cur_classes = g_test_set[:,4], np.zeros(75)

    for idx,x in enumerate(g_test_set[:,0:4]):
        for cls,c in enumerate(g_centroids):
            distances[cls][idx]=vec_distance(x,c)
        # 每个点属于最近的聚类中心
        min_distance = min(distances[:,idx])
        for cls,c in enumerate(g_centroids):
            cur_classes[idx] = cls if min_distance == distances[cls][idx] else cur_classes[idx]
    for idx,x in enumerate(cur_classes):
        cls = int(target_classes[idx])
        if cur_classes[idx] == target_classes[idx]:
            success[cls] = success[cls] + 1
        else:
            mistake[cls] = mistake[cls] + 1
    print success,mistake
    pass

def kmeans_train():
    global g_train_set
    global g_centroids

    # 欧式距离向量，大小3x25
    distances = np.zeros((3,75))
    # 初始聚类中心
    #g_centroids = np.array([g_train_set[1:2,0:4],g_train_set[50:51,0:4],g_train_set[59:60,0:4]])
    g_centroids = np.array([g_train_set[0:1,0:4],g_train_set[25:26,0:4],g_train_set[50:51,0:4]])
    old_centroids = g_centroids
    # 目标分类向量，当前分类向量
    target_classes, cur_classes = g_train_set[:,4], np.zeros(75)

    print "first g_centroids", g_centroids[0],g_centroids[1],g_centroids[2]

    # 迭代，直到聚类中心点不再变化
    while True:
        # 聚类中所有点的步长和的向量
        step_sum = np.zeros((3,4))
        # 当前聚类中所有点的计数
        step_count = np.zeros(3)

        # 计算距离向量，聚类中心到每个点的距离
        for idx,x in enumerate(g_train_set[:,0:4]):
            for cls,c in enumerate(g_centroids):
                distances[cls][idx]=vec_distance(x,c)

            # 每个点属于最近的聚类中心
            min_distance = min(distances[:,idx])
            for cls,c in enumerate(g_centroids):
                cur_classes[idx] = cls if min_distance == distances[cls][idx] else cur_classes[idx]

        # 重新聚类, 寻找新的聚类中心点：当前聚类中所有点的均值, x可能的取值是（0,1,2）代表所在行的聚类
        for idx,cls in enumerate(cur_classes):
            x = int(cls)
            #print idx, x, g_train_set[idx,0:4], step_sum[x]
            step_sum[x] = step_sum[x] + g_train_set[idx,0:4]
            step_count[x] = step_count[x] + 1
        for cls,c in enumerate(g_centroids):
            g_centroids[cls] = step_sum[cls]/step_count[cls]
        print g_centroids[0],g_centroids[1],g_centroids[2]

        if np.array_equal(old_centroids,g_centroids):
            # 聚类中心不再变化，收敛完成
            #kmeans_plot3d()
            break
        else:
            old_centroids = g_centroids
    pass

def kmeans_init():
    global g_train_set
    global g_test_set

    datafile = Config.DATA_FILE
    data = data_import(datafile,',')

    # 读取并合并训练集数组
    g_train_set = np.hstack((data[0:25,0:4],np.full((25,1),0)))
    g_train_set = np.vstack((g_train_set,np.hstack((data[50:75  ,0:4],np.full((25,1),1)))))
    g_train_set = np.vstack((g_train_set,np.hstack((data[100:125,0:4],np.full((25,1),2)))))

    # 读取并合并测试集数组
    g_test_set = np.hstack((data[25:50,0:4],np.full((25,1),0)))
    g_test_set = np.vstack((g_test_set,np.hstack((data[75:100  ,0:4],np.full((25,1),1)))))
    g_test_set = np.vstack((g_test_set,np.hstack((data[125:150,0:4],np.full((25,1),2)))))

def kmeans_plot():
    global g_train_set
    global g_test_set
    global g_centroids

    # 以0,1,3特征绘制3D图
    my_set = g_train_set;

    plot.scatter(my_set[0 :25,0],my_set[0: 25,1],marker="x",color="k")
    plot.scatter(my_set[25:50,0],my_set[25:50,1],marker="x",color="m")
    plot.scatter(my_set[50:75,0],my_set[50:75,1],marker="x",color="c")
    plot.scatter(my_set[0 :25,2],my_set[0: 25,3],marker="o",color="k")
    plot.scatter(my_set[25:50,2],my_set[25:50,3],marker="o",color="m")
    plot.scatter(my_set[50:75,2],my_set[50:75,3],marker="o",color="c")

    my_set = g_test_set;

    plot.scatter(my_set[0 :25,0],my_set[0: 25,1],marker="x",color="k")
    plot.scatter(my_set[25:50,0],my_set[25:50,1],marker="x",color="m")
    plot.scatter(my_set[50:75,0],my_set[50:75,1],marker="x",color="c")
    plot.scatter(my_set[0 :25,2],my_set[0: 25,3],marker="o",color="k")
    plot.scatter(my_set[25:50,2],my_set[25:50,3],marker="o",color="m")
    plot.scatter(my_set[50:75,2],my_set[50:75,3],marker="o",color="c")

    plot.xlabel(Config.LABEL_NAMES[0]+"/"+Config.LABEL_NAMES[2])
    plot.ylabel(Config.LABEL_NAMES[1]+"/"+Config.LABEL_NAMES[3])

    plot.autoscale(tight=True)
    plot.grid()
    plot.show()

def kmeans_plot3d():
    global g_train_set
    global g_test_set
    global g_centroids

    # 以0,1,3特征绘制3D图
    my_set = g_train_set;

    fig = plot.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(my_set[0 :25,0],my_set[0: 25,2],my_set[0 :25,3],marker="o",c="r",edgecolor="r")
    ax.scatter(my_set[25:50,0],my_set[25:50,2],my_set[25:50,3],marker="o",c="c",edgecolor="c")
    ax.scatter(my_set[50:75,0],my_set[50:75,2],my_set[50:75,3],marker="o",c="b",edgecolor="b")
    # 聚类中心
    ax.set_xlabel(Config.LABEL_NAMES[0])
    ax.set_ylabel(Config.LABEL_NAMES[1])
    ax.set_zlabel(Config.LABEL_NAMES[3])
    ax.plot(g_centroids[0][:,0],g_centroids[0][:,2],g_centroids[0][:,3],marker="^",c="c")
    ax.plot(g_centroids[1][:,0],g_centroids[1][:,2],g_centroids[1][:,3],marker="^",c="b")
    ax.plot(g_centroids[2][:,0],g_centroids[2][:,2],g_centroids[2][:,3],marker="^",c="r")
    plot.autoscale(tight=True)
    plot.grid()
    plot.show()
if __name__ == "__main__":
    # test_vec_distance()
    # exit()
    # 数据导入
    kmeans_init()
    #kmeans_plot()
    # 训练数据，k-means迭代
    kmeans_train()
    # 检验测试集
    kmeans_test()

#!/usr/bin/python
# -*- coding: utf-8 *-*

#
# 对校准系数进行拟合，得到校准方程
#
import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
    DATA_FILE="./regulate2.txt"


def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

if __name__ == "__main__":
    datafile = Config.DATA_FILE
    data = data_import(datafile,' ')

    x = data[:,1]
    y = data[:,0]
    print("lenx leny:%d %d" %(len(x), len(y)))

    plot.xlabel("adc")
    plot.ylabel("mm")
    plot.autoscale(tight=True)
    plot.grid()
    plot.scatter(x,y)
    plot.autoscale(tight=True)

    degs = [1]
    for deg in degs:
        fit = np.polyfit(x,y,deg)
        fnd = np.poly1d(fit)
        fx = np.linspace(0,x[-1],10)
        plot.plot(fx,fnd(fx))
        print fit

    for idx in range(len(x)):
        calc = fit[0]*x[idx]+fit[1]
        print ("%-.08f\t %.08f\t %.08f\t %.08f" % (x[idx] , calc , y[idx] , calc-y[idx]))

    plot.show()

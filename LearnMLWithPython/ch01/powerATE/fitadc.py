#!/usr/bin/python
# -*- coding: utf-8 *-*

#
# 对校准系数进行拟合，得到校准方程
#
import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
    DATA_FILE="./regulate1.txt"


def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

if __name__ == "__main__":
    datafile = Config.DATA_FILE
    data = data_import(datafile,' ')

    mmVals = data[:,1]
    adcVals = data[:,0]
    print("lenADC lenMM:%d %d" %(len(adcVals),len(mmVals)))

    plot.xlabel("adc")
    plot.ylabel("mm")
    plot.autoscale(tight=True)
    plot.grid()
    plot.scatter(adcVals,mmVals)
    plot.autoscale(tight=True)

    degs = [1]
    for deg in degs:
        fit = np.polyfit(adcVals,mmVals,deg)
        fnd = np.poly1d(fit)
        fx = np.linspace(0,adcVals[-1],10)
        plot.plot(fx,fnd(fx))
        print fit

    print ("%-13s\t %-13s\t %-13s\t %-13s\t %-13s" % ("adc测量值" , "方程值" , "万用表值" , "残差", "残差偏差千分比"))

    for idx in range(len(adcVals)):
        calc = fit[0]*adcVals[idx]+fit[1]
        diff = calc-mmVals[idx]
        diffPercent = 0
        if(adcVals[idx] != 0):
            diffPercent = (1000*diff)/adcVals[idx]
        print ("%-.08f\t %.08f\t %.08f\t %.08f\t %.08f" % (adcVals[idx] , calc , mmVals[idx] , diff, diffPercent))

    plot.show()

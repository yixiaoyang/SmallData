#!/usr/bin/python
# -*- coding: utf-8 *-*

import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
    DATA_FILE="./regulate1.txt"

def data_import(file,delimiter):
    return np.genfromtxt(file,delimiter=delimiter)

if __name__ == "__main__":
    # 1. import data
    #basedir = os.path.abspath(os.path.dirname(__file__))
    datafile = Config.DATA_FILE
    data = data_import(datafile,'\t')

    # 2. clean data
    if len(data) < 10:
        exit()
    y = data[:,0]
    x = data[:,1]
    print("lenx leny:%d %d" %(len(x), len(y)))


    # 3. plot data
    #plot.scatter(x,y)
    plot.xlabel("count")
    plot.ylabel("value")
    #plot.xticks([w*7*24 for w in range(10)], ["week %i" %w for w in range(10)])
    plot.autoscale(tight=True)
    plot.grid()

    splitCell = 30
    offset = 0
    while offset < len(x):
        xsplit = x[offset:offset+splitCell]
        ysplit = y[offset:offset+splitCell]
        offset += splitCell
        print np.mean(xsplit),np.mean(ysplit)

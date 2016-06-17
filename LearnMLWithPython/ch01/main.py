#!/usr/bin/python
# -*- coding: utf-8 *-*

import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
	DATA_FILE="data/web_traffic.tsv"

# >>> data=numpy.genfromtxt("/devel/git/github/SmallData/LearnMLWithPython/ch01/data/web_traffic.tsv",delimiter='\t')
# >>> data
# array([[  1.00000000e+00,   2.27200000e+03],
#        [  2.00000000e+00,              nan],
#        [  3.00000000e+00,   1.38600000e+03],
#        ...,
#        [  7.41000000e+02,   5.39200000e+03],
#        [  7.42000000e+02,   5.90600000e+03],
#        [  7.43000000e+02,   4.88100000e+03]])
# >>> len(data)
# 743
def data_import(file,delimiter):
	return np.genfromtxt(file,delimiter=delimiter)

def data_clean(data):
	pass

def modal_training(data):
	pass

def modal_test(data):
	pass

def modal_run(data):
	pass

def data_plot():
	pass

def data_print():
	pass

if __name__ == "__main__":
	# 1. import data
	basedir = os.path.abspath(os.path.dirname(__file__))
	datafile = basedir + "/data/web_traffic.tsv"
	data = data_import(datafile,'\t')

	# 2. clean data
	if len(data) < 10:
		exit()
	x = data[:,0]
	y = data[:,1]
	print("nan count:%d" % np.sum(np.isnan(y)))

	x = x[~np.isnan(y)]
	y = y[~np.isnan(y)]
	print("lenx leny:%d %d" %(len(x), len(y)))

	# 3. plot data
	plot.scatter(x,y)
	plot.xlabel("Time")
	plot.ylabel("Hits/Hour")
	plot.xticks([w*7*24 for w in range(10)], ["week %i" %w for w in range(10)])
	plot.autoscale(tight=True)
	plot.grid()


	# 4. ploy fit deg=1
	degs = [1,2,3,4,10,100]
	for deg in degs:
		fit = np.polyfit(x,y,deg)
		fnd = np.poly1d(fit)
		fx = np.linspace(0,x[-1],1000)
		plot.plot(fx,fnd(fx),linewidth=4)

	plot.legend(["deg=%i"%d for d in degs],loc="upper left")

	plot.show()

#!/usr/bin/python
# -*- coding: utf-8 *-*

import os
import numpy as np
import matplotlib.pyplot as plot

class Config:
	DATA_FILE="/devel/server/nfsroot/rootfs/opt/PowerAte/config/data-ch2/regulate2.dat"


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


	# 4. ploy fit deg=1
	degs = [1]
	#for deg in degs:
		#fit = np.polyfit(x,y,deg)
		#fnd = np.poly1d(fit)
		#fx = np.linspace(0,x[-1],0.1)
		#plot.plot(fx,fnd(fx))
	#	plot.plot(fit)
	
	plot.scatter([n for n in range(len(x))],x, color='blue', label='mm')
	plot.scatter([n for n in range(len(y))],y, color='red',  label='adc')
	plot.legend()
    
	#plot.legend(["deg=%i"%d for d in degs],loc="upper left")

	plot.show()

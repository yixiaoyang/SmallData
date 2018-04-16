#!/usr/bin/python
# -*- coding: utf-8 *-*

import datetime
import os
import math
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

def progressbar(cur,total):
    percent = '{:.2%}'.format( float(cur)/total)
    sys.stdout.write('\r')
    sys.stdout.write('[%-50s] %s' % ( '=' * int(math.floor(cur * 50 /total)),percent))
    sys.stdout.flush()
    if cur == total:
        sys.stdout.write('\n')

def convert_data(srcFile, dstFile, interal=2):
    if not os.path.exists(srcFile):
        return
    readCount, writeCount = 0, 0
    ac_types = {
        "NONE": "0",
        "A1": "1",
        "A2": "2",
        "A3": "3"
    }

    dfp = open(dstFile, 'w')
    with open(srcFile, 'r') as fp:
        header = fp.readline()
        header = header.replace("\r\n", "")
        header = header + ",weekend,workday,t0,t1,t2,t3\r\n"
        dfp.write(header)

        for line in fp.readlines():
            readCount += 1

            values = line.split(',')
            ac_time = ""
            try:
                ac_time = datetime.datetime.strptime(values[0], "%d/%m/%Y %H:%M")
            except Exception, e1:
                try:
                    ac_time = datetime.datetime.strptime(values[0], "%H:%M %d/%m/%Y")
                except Exception, e2:
                    continue

            strWeek = "0,0"
            strTime = "0,0,0,0"
            if ac_time.weekday() >= 5:
                strWeek = "1,0"
            else:
                strWeek = "0,1"
            if ac_time.hour <= 6 or ac_time.hour >= 22:
                strTime = "1,0,0,0"
            elif ac_time.hour >= 6 and ac_time.hour <= 10:
                strTime = "0,1,0,0"
            elif ac_time.hour >= 10 and ac_time.hour <= 16:
                strTime = "0,0,1,0"
            elif ac_time.hour >= 16 and ac_time.hour <= 22:
                strTime = "0,0,0,1"

            # convert ac_type
            values[2] = ac_types[values[2]]
            line = ','.join(values[1:])
            line = line.replace("\r\n", "")
            newLine = line + "," + strWeek + "," + strTime + "\r\n"
            dfp.write(newLine)
            writeCount += 1
        fp.close()
    dfp.close()

    if readCount != writeCount:
        print("Error:read data failed")
    print("readCount:%d, writeCount:%d" % (readCount, writeCount))


class RecurrentNeuralNetwork:
    def __init__(self):
        self.session = tf.Session()
        self.inputs = None
        self.input_layer = None
        self.label_layer = None
        self.weights = None
        self.biases = None
        self.lstm_cell = None
        self.prediction = None
        self.loss = None
        self.trainer = None

    def __del__(self):
        self.session.close()

    def train(self, train_x, train_y, learning_rate=0.02, epochs=1, batch_n=1, input_n=1, hidden_n=4):
        seq_n = len(train_x)
        input_n = len(train_x[0])
        output_n = len(train_y[0])

        # self.input_layer = tf.placeholder(tf.float32, in_shape)
        self.inputs = tf.placeholder(tf.float32, [batch_n, input_n])
        self.label_layer = tf.placeholder(tf.float32, [output_n])
        self.input_layer = [tf.reshape(i, (1, input_n)) for i in tf.split(0, batch_n, self.inputs)]

        self.weights = tf.Variable(tf.random_normal([hidden_n, output_n]))
        self.biases = tf.Variable(tf.random_normal([output_n]))
        self.lstm_cell = rnn_cell.BasicLSTMCell(hidden_n, forget_bias=1.0)
        
        outputs, states = rnn.rnn(self.lstm_cell, self.input_layer, dtype=tf.float32)
        self.prediction = tf.matmul(outputs[-1], self.weights) + self.biases
        #self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        #self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.label_layer))
        self.trainer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.loss)


        initer = tf.global_variables_initializer()
        writer = tf.train.SummaryWriter("./graph-rnn", self.session.graph)

        tf.scalar_summary("loss", self.loss)
        tf.scalar_summary("prediction", self.prediction[0][0])
        #tf.scalar_summary("label", self.label_layer[0])
        #tf.scalar_summary("input0", self.input_layer[0][0][0])
        merged_summary = tf.merge_all_summaries()

        self.session.run(initer)

        total_seq = seq_n - batch_n
        for epoch in range(epochs):
            for idx in range(0, total_seq):
                input_x = train_x[idx:idx + batch_n]
                output_y = train_y[idx]
                feed_dict = {self.inputs: input_x, self.label_layer: output_y}
                _, summary = self.session.run([self.trainer, merged_summary], feed_dict=feed_dict)
                
                #progressbar(idx+1, seq_n - batch_n)
                writer.add_summary(summary, idx+epoch*total_seq)

    def predict(self, test_x, test_y, batch_n):
        seq_n = len(test_x)
        input_n = len(test_x[0])
        
        acc_predict_cnt, acc_cnt = 0,0
        no_acc_predict_cnt, no_acc_cnt = 0,0

        for idx in range(batch_n, seq_n - batch_n - 1):
            input_x = test_x[idx:idx + batch_n]
            label_y = test_y[idx]
            predict_y = self.session.run(self.prediction, feed_dict={self.inputs: input_x})

            if idx < 64+batch_n:
                print("line %d:%f %f" % (idx, label_y, predict_y))

            if label_y >= 1.0:
                acc_cnt += 1
                if abs(label_y - predict_y) < 0.2:
                    acc_predict_cnt += 1
            else:
                no_acc_cnt += 1
                if abs(label_y - predict_y) < 0.2:
                    no_acc_predict_cnt += 1


        # 有事故，预测成功的准确率
        acc_accuracy = float(acc_predict_cnt)/acc_cnt
        no_acc_accuracy = float(no_acc_predict_cnt)/no_acc_cnt

        # 无事故，预测成功的准确率
        print("no_acc_predict_cnt=%d, acc_predict_cnt=%d"%(no_acc_cnt, acc_cnt))
        print("predict no_acc_predict_cnt=%d, acc_predict_cnt=%d"%(no_acc_predict_cnt, acc_predict_cnt))
        print("acc accuracy= %f"% acc_accuracy)
        print("no acc accuracy= %f"% no_acc_accuracy)

    def test(self, train_x, train_y, test_x, test_y, batch_n, epochs):
        self.train(train_x, train_y, batch_n=batch_n, epochs=epochs)
        self.predict(test_x, test_y, batch_n=batch_n)

    def test(self, train_x, train_y, test_x, test_y, batch_n, epochs):
        self.train(train_x, train_y, batch_n=batch_n, epochs=epochs)
        
        self.predict(train_x, train_y, batch_n=batch_n)
        #self.predict(test_x, test_y, batch_n=batch_n)


'''
    data_import
        dtype =[
                        #('time'                ,'<S32'),     #     time
                        ('ac_num'               ,int),        # 0   ac_num
                        ('ac_type'              ,int),        # 1   ac_type
                        ('holiday'              ,int),        # 2   holiday
                        ('prec'                 ,float),      # 3   prec
                        ('visibility'           ,int),        # 4   visibility  (0-?)
                        ('wind'                 ,float),      # 5   wind        (0-1x?)
                        ('wind_dir'             ,int),        # 6   wind_dir    (0-360)
                        ('fog'                  ,int),        # 7   fog
                        ('rain'                 ,int),        # 8   rain
                        ('sun_rise'             ,int),        # 9   sun_rise
                        ('sun_set'              ,int),        # 10  sun_set
                        ('weekend'              ,int),        # 11  weekend
                        ('workday'              ,int),        # 12  workday
                        ('t0'                   ,int),        # 13  t0
                        ('t1'                   ,int),        # 14  t1
                        ('t2'                   ,int),        # 15  t2
                        ('t3'                   ,int),        # 16  t3
        ]
'''
def normalize(x):
    return (x-min(x))/(max(x)-min(x))

def data_import(file, delimiter=','):
    x_cols = (1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    y_cols = (0)
    
    x = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=x_cols)
    # visibility 4
    x[:,3] = normalize(x[:,3])
    # wind 5
    x[:,4] = normalize(x[:,4])
    # wind_dir, ignore
    # x[:,5] = normalize(x[:,5])

    y = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=y_cols)
    y = np.array([[value] for value in y])
    return x, y

if __name__ == "__main__":
    # convert_data("./data/4hours.csv", "./data/4hours2.csv")
    # convert_data("./data/2hours.csv", "./data/2hours2.csv")
    train_x, train_y = data_import("./data/4hours-training.csv")
    test_x, test_y = data_import("./data/4hours-test.csv")
    print("training shape: x%s, y%s" % (str(train_x.shape), str(train_y.shape)))
    print("test shape: x%s, y%s" % (str(test_x.shape), str(test_y.shape)))
    print("type(train_x)=%s" % (type(train_x)))
    #print(train_x[0:20,])
    nn = RecurrentNeuralNetwork()
    nn.test(train_x, train_y, test_x, test_y, batch_n=1, epochs=8)
    

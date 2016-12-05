#!/usr/bin/python
# -*- coding: utf-8 *-*

import numpy as np
import datetime
import os

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


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

    def train(self, train_x, train_y, learning_rate=0.05, epochs=1, batch_n=1, input_n=1, hidden_n=4):
        '''
        train_x     输入数据为 999 x 16
        batch_n     即batch_n=4行为一个批量seq长度，input_n为宽度的数据为一个输入
                    input:
                        [a1 a2.. a15]
                        [b1 b2.. b15]
                        [c1 c2.. c15]
                        [d1 d2.. d15] 
                    out:[d0] 输出即为下一个时间段交通事故的预测数目
        hidden_n    隐藏层特征数
        seq_len
        '''
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
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.label_layer))
        self.trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

        initer = tf.global_variables_initializer()
        writer = tf.train.SummaryWriter("./graph", self.session.graph)

        tf.scalar_summary("loss", self.loss)
        tf.scalar_summary("prediction", self.prediction[0][0])
        tf.scalar_summary("label", self.label_layer[0])
        merged_summary = tf.merge_all_summaries()

        self.session.run(initer)
        for epoch in range(epochs):
            for idx in range(0, seq_n - batch_n):
                input_x = train_x[idx:idx + batch_n]
                output_y = train_y[idx]
                feed_dict = {self.inputs: input_x, self.label_layer: output_y}
                _, summary = self.session.run([self.trainer, merged_summary], feed_dict=feed_dict)
                writer.add_summary(summary, idx)

    def predict(self, test_x, test_y, batch_n):
        seq_n = len(test_x)
        input_n = len(test_x[0])
        for idx in range(batch_n, seq_n - batch_n - 1):
            input_x = test_x[idx:idx + batch_n]
            label_y = test_y[idx]
            predict_y = self.session.run(self.prediction, feed_dict={self.inputs: input_x})
            print("line %d:%f %f" % (idx, label_y, predict_y))

    def test(self, train_x, train_y, test_x, test_y, batch_n, epochs):
        self.train(train_x, train_y, batch_n=batch_n, epochs=epochs)
        self.predict(test_x, test_y, batch_n=batch_n)


'''
    data_import
        dtype =[
                        #('time'                ,'<S32'),     #     time
                        ('ac_num'               ,int),        # 0   ac_num
                        ('ac_type'              ,int),        # 1   ac_type
                        ('holiday'              ,int),        # 2   holiday
                        ('prec'                 ,float),      # 3   prec
                        ('visibility'           ,int),        # 4   visibility
                        ('wind'                 ,float),      # 5   wind
                        ('wind_dir'             ,int),        # 6   wind_dir
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
def data_import(file, delimiter=','):
    x_cols = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
    y_cols = (0)
    x = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=x_cols)
    y = np.genfromtxt(file, delimiter=delimiter, skip_header=True, usecols=y_cols)
    y = np.array([[value] for value in y])
    return x, y


if __name__ == "__main__":
    # convert_data("./data/4hours.csv", "./data/4hours2.csv")
    # convert_data("./data/2hours.csv", "./data/2hours2.csv")

    train_x, train_y = train_data = data_import("./data/4hours-training.csv")
    test_x, test_y = train_data = data_import("./data/4hours-test.csv")
    print("training shape: x%s, y%s" % (str(train_x.shape), str(train_y.shape)))
    print("test shape: x%s, y%s" % (str(test_x.shape), str(test_y.shape)))
    print("type(train_x)=%s" % (type(train_x)))

    nn = RecurrentNeuralNetwork()
    nn.test(train_x, train_y, test_x, test_y, batch_n=5, epochs=1)

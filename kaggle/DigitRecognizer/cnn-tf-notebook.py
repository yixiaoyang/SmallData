# coding: utf-8

# ### Packages and imports

# In[1]:

#!/usr/bin/python


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

#get_ipython().magic(u'matplotlib inline')


# ### Simulation Constans
#
# 卷积计算过程：
# 0. 输入： 28x28
# 1. 卷积核5x5，输入1, 特征32个 卷积输出：32@24x24
# 2. 池化2x2，池化输出：32@12x12
# 3. 卷积核5x5, 输出64@8x8
# 4. 池化2x2，池化输出：64@4x4
# 5. tanh归一化，得到64x4x4个输入
# 6. 神经元全连接MLP（multi-layer perceptrons）

# In[2]:

'''
@func       Computes a 2-D convolution given 4-D input and filter tensors.
@param      input   4-D input tensor of shape [batch, in_height, in_width, in_channels]
            filter  4-D filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
@return
@note       卷积运算
'''
def conv2d(input,filter):
    return tf.nn.conv2d(input, filter, strides=[1,1,1,1], padding='SAME')


# In[3]:

'''
@func       Performs the max pooling on the input.
@param      input   4-D Tensor with shape [batch, height, width, channels] and type tf.float32
            ksize   A list of ints that has length >= 4. The size of the window for each dimension of the input tensor.
            strides A list of ints that has length >= 4. The stride of the sliding window for each dimension of the input tensor
@return
@note       最大池化运算
'''
def max_pool_2x2(input):
    return tf.nn.max_pool(input, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")


# In[4]:

'''
@func       outputs random values from a truncated normal distribution.
'''
def init_w(shape):
    # the standard deviation is 0.1
    value = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(value)


# In[5]:

'''
@func       outputs random values as bias
'''
def init_b(shape):
    value = tf.constant(0.1, shape=shape)
    return tf.Variable(value)


# In[6]:

'''
@class  构建CNN网络
'''
session = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None,784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

# x @28x28 + filter 32 features @5x5 => 32@24x24 => 32@12x12
w_conv1 = init_w([5,5,1,32])
b_conv1 = init_b([32])
# use RELU as active function
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 32@12x12 + filter 64 features @5x5 => 64@8x8 => 64@4x4
w_conv2 = init_w([5,5,32,64])
b_conv2 = init_b([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# MPL Densely Connected Layer
w_fc1 = init_w([7*7*64,1024])
b_fc1 = init_b([1024])
hpool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(hpool2_flat, w_fc1) + b_fc1)


# In[7]:

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Readout Layer
w_fc2 = init_w([1024,10])
b_fc2 = init_w([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
yy = tf.placeholder(tf.float32,[None,10])

# model
cross_entropy = tf.reduce_mean(-tf.reduce_sum(yy * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = y_conv
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(yy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# preparing data
#print("downloading data...")
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#print("training data...")

# train
session.run(tf.global_variables_initializer())
for i in range(400):
    batch = mnist.train.next_batch(50)

    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], yy: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    feed_dict = {x: batch[0], yy: batch[1], keep_prob: 0.5}
    train_step.run(feed_dict = feed_dict)


# In[ ]:

# ERROR: bad alloc, why?
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, yy: mnist.test.labels, keep_prob: 1.0}))
#test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, yy: mnist.test.labels})
#print("test_accuracy ")
#print(test_accuracy)

print("read data")
test = pd.read_csv('./data/test.csv')

print("predict")
# Convert the dataframe to a numpy array
test_data = StandardScaler().fit_transform(np.float32(test.values))
 # Reshape the data into 42000 2d images
test_data = test_data.reshape(-1, 28, 28, 1)
test_pred = session.run(prediction, feed_dict={x_image:test_data})
test_labels = np.argmax(test_pred, axis=1)

print("plot")
k = 0 # Try different image indices k
print("Label Prediction: %i"%test_labels[k])
fig = plt.figure(figsize=(2,2)); plt.axis('off')
plt.imshow(test_data[k,:,:,0]); plt.show()

print ("done")
# clean
#session.close()

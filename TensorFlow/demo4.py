import tensorflow as tf    
from tensorflow.python.ops.rnn_cell import MultiRNNCell, BasicLSTMCell, LSTMCell    
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

if __name__ == '__main__':
	np.random.seed(1)      
	size = 3
	batch_seq_len= 4
	n_steps = 5
	seq_width = 6

	initializer = tf.random_uniform_initializer(-1,1) 

	#sequence we will provide at runtime  
	seq_input = tf.placeholder(tf.float32, [n_steps, batch_seq_len, seq_width])
	
	#what timestep we want to stop at
	early_stop = tf.placeholder(tf.int32)
	
	#inputs for rnn needs to be a list, each item being a timestep. 
	#we need to split our input into each timestep, and reshape it because split keeps dims by default  
	inputs = [tf.reshape(i, (batch_seq_len, seq_width)) for i in tf.split(0, n_steps, seq_input)]

	cell = LSTMCell(size, seq_width, initializer=initializer)

	initial_state = cell.zero_state(batch_seq_len, tf.float32)
	outputs, states = rnn.rnn(cell, inputs, initial_state=initial_state, sequence_length=early_stop)
	
	#set up lstm
	iop = tf.global_variables_initializer()
	
	#create initialize op, this needs to be run by the session!
	session = tf.Session()
	session.run(iop)
	
	#actually initialize, if you don't do this you get errors about uninitialized stuff
	# 4 X 10 X 5
	feed = {early_stop:5, seq_input:np.random.rand(n_steps, batch_seq_len, seq_width).astype('float32')}
	
	#define our feeds. 
	#early_stop can be varied, but seq_input needs to match the shape that was defined earlier
	outs = session.run(outputs, feed_dict=feed)
	
	#run once
	#output is a list, each item being a single timestep. Items at t>early_stop are all 0s
	for i in range(len(outs)):
		print i, outs[i]
	print len(outs)


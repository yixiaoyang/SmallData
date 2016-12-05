#!/usr/bin/python
# -*- coding: utf-8 *-*

import tensorflow as tf

def demo1():
    state = tf.Variable(0,name="counter")
    one = tf.constant(1)
    new_value = tf.add(state, one)
    update = tf.assign(state, new_value)

    init_op = tf.global_variables_initializer()

    with tf.Session() as session:
        session.run(init_op)
        print session.run(state)

        # run 3 times:
        for _ in range(10):
            session.run(update)
            print session.run(state)

if __name__ == "__main__":
    demo1()

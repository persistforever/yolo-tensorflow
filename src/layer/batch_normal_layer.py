# -*- coding=utf8 -*-
'''
========================================================
  Copyright (C) 2017 All rights reserved.

  filename : batch_normal.py
  author   : lihongwei / lhw446@qq.com
  date     : 2017-12-27
  desc     : batch normalization layer
========================================================
'''
import tensorflow as tf


class BatchNormalLayer:

    def __init__(self, shape, decay=0.9, epsilon=1e-5, name='bn'):
        # params
        self.decay = decay
        self.epsilon = epsilon
        self.name = name

        with tf.name_scope('%s_bn' % (self.name)) as scope:
            self.scale = tf.Variable(tf.ones([shape]), dtype=tf.float32, name='scale')
            self.beta = tf.Variable(tf.zeros([shape]), dtype=tf.float32, name='beta')
            self.pop_mean = tf.Variable(tf.zeros([shape]), dtype=tf.float32, trainable=False, name='mean')
            self.pop_var = tf.Variable(tf.ones([shape]), dtype=tf.float32, trainable=False, name='varance')

    def get_output(self, inputs, is_training):

        def _bn_train():
            batch_mean, batch_var = tf.nn.moments(inputs, axes=[0,1,2], name='moments')
            assign_mean = tf.assign(self.pop_mean, 
                self.pop_mean * self.decay + batch_mean * (1.0 - self.decay))
            assign_variance = tf.assign(self.pop_var, 
                self.pop_var * self.decay + batch_var * (1.0 - self.decay))
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, self.beta, self.scale, self.epsilon)
            
        def _bn_test():
            return tf.nn.batch_normalization(inputs,
                self.pop_mean, self.pop_var, self.beta, self.scale, self.epsilon)
    
        output = tf.cond(is_training, _bn_train, _bn_test)    
        return output

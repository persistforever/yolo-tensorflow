# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class PoolLayer:
    
    def __init__(self, input_shape, n_size=2, stride=2, mode='max', 
                 resp_normal=False, name='pool'):
        # params
        self.input_shape = input_shape
        self.n_size = n_size
        self.stride = stride
        self.mode = mode
        self.resp_normal = resp_normal
        self.name = name
        
    def get_output(self, input):
        # calculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], int(self.input_shape[1]/self.stride),
                             int(self.input_shape[2]/self.stride), self.input_shape[3]]
        
        if self.mode == 'max':
            self.pool = tf.nn.max_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding='SAME')
        elif self.mode == 'avg':
            self.pool = tf.nn.avg_pool(
                value=input, ksize=[1, self.n_size, self.n_size, 1],
                strides=[1, self.stride, self.stride, 1], padding='SAME')
        if self.resp_normal:
            self.hidden = tf.nn.local_response_normalization(
                self.pool, depth_radius=7, alpha=0.001, beta=0.75)
        else:
            self.hidden = self.pool
        self.output = self.hidden
        
        # 打印网络权重、输入、输出信息
        print('%-10s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '(%d * %d / %d)' % (
                self.n_size, self.n_size, self.stride),
            '(%d * %d * %d)' % (
                self.input_shape[1], self.input_shape[2], self.input_shape[3]),
            '(%d * %d * %d)' % (
                self.output_shape[1], self.output_shape[2], self.output_shape[3])))
        
        return self.output
# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf
import src.layer.utils as utils


class PoolLayer:
    
    def __init__(self, y_size, x_size, y_stride, x_stride, mode='max', 
                 resp_normal=False, name='pool',
                 input_shape=None, prev_layer=None):
        # params
        self.y_size = y_size
        self.x_size = x_size
        self.y_stride = y_stride
        self.x_stride = x_stride
        self.mode = mode
        self.resp_normal = resp_normal
        self.name = name
        self.ltype = 'pool'
        if prev_layer:
            self.input_shape = prev_layer.output_shape
            self.prev_layer = prev_layer
        elif input_shape:
            self.input_shape = input_shape
            self.prev_layer = None
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 计算感受野
        self.feel_field = [1, 1]
        self.feel_field[0] = min(self.input_shape[0], 1 * int(self.y_size))
        self.feel_field[1] = min(self.input_shape[1], 1 * int(self.x_size))
        prev_layer = self.prev_layer
        while prev_layer:
            if prev_layer.ltype == 'conv':
                self.feel_field[0] = min(prev_layer.input_shape[0], 
                    self.feel_field[0] + int((prev_layer.y_size+1)/2))
                self.feel_field[1] = min(prev_layer.input_shape[1], 
                    self.feel_field[1] + int((prev_layer.x_size+1)/2))
            elif prev_layer.ltype == 'pool':
                self.feel_field[0] = min(prev_layer.input_shape[0], 
                    self.feel_field[0] * int(prev_layer.y_size))
                self.feel_field[1] = min(prev_layer.input_shape[1], 
                    self.feel_field[1] * int(prev_layer.x_size))
            prev_layer = prev_layer.prev_layer
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [
            int(self.input_shape[0]/self.y_stride),
            int(self.input_shape[1]/self.x_stride), 
            self.input_shape[2]]
        print('%-10s\t%-25s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '((%d, %d) / (%d, %d))' % (
                self.y_size, self.x_size, self.y_stride, self.x_stride),
            '(%d, %d, %d)' % (
                self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            '(%d, %d, %d)' % (
                self.output_shape[0], self.output_shape[1], self.output_shape[2]),
            '(%d, %d)' % (
                self.feel_field[0], self.feel_field[1])))
        self.calculation = self.output_shape[0] * self.output_shape[1] * \
            self.output_shape[2] * self.y_size * self.x_size
        
    def get_output(self, input, is_training=True):
        with tf.name_scope('%s_cal' % (self.name)) as scope: 
            if self.mode == 'max':
                self.pool = tf.nn.max_pool(
                    value=input, ksize=[1, self.y_size, self.x_size, 1],
                    strides=[1, self.y_stride, self.x_stride, 1], padding='SAME', name='maxpool')
            elif self.mode == 'avg':
                self.pool = tf.nn.avg_pool(
                    value=input, ksize=[1, self.y_size, self.x_size, 1],
                    strides=[1, self.y_stride, self.x_stride, 1], padding='SAME', name='avgpool')
            if self.resp_normal:
                self.hidden = tf.nn.local_response_normalization(
                    self.pool, depth_radius=7, alpha=0.001, beta=0.75, name='lrn')
            else:
                self.hidden = self.pool
            self.output = self.hidden
        
        return self.output

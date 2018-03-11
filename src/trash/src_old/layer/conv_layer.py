# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import math
import tensorflow as tf
import random


class ConvLayer:
    
    def __init__(self, input_shape, n_size, n_filter, stride=1, activation='relu',
                 batch_normal=False, weight_decay=None, name='conv'):
        # params
        self.input_shape = input_shape
        self.n_size = n_size
        self.n_filter = n_filter
        self.activation = activation
        self.stride = stride
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        self.name = name
        self.ltype = 'conv'
        
        self.leaky_scale = tf.constant(0.1, dtype=tf.float32)
    
        with tf.name_scope('%s_def' % (self.name)) as scope:
            # 权重矩阵
            scale = math.sqrt(2.0 / (self.n_size * self.n_size * self.input_shape[3]))
            init_value = scale * numpy.random.normal(size=[
                self.n_size, self.n_size, self.input_shape[3], self.n_filter], loc=0.0, scale=1.0)
            self.weight = tf.Variable(init_value, dtype=tf.float32, name='weight')
            
            # 偏置向量
            self.bias = tf.Variable(
                initial_value=tf.constant(0.0, shape=[self.n_filter]),
                name='bias')
        
            # batch normalization 技术的参数
            if self.batch_normal:
                self.epsilon = 1e-5
                self.gamma = tf.Variable(
                    initial_value=tf.constant(1.0, shape=[self.n_filter]),
                    name='gamma')
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], int(self.input_shape[1]/self.stride),
                             int(self.input_shape[2]/self.stride), self.n_filter]
        print('%-10s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '(%d * %d * %d / %d)' % (
                self.n_size, self.n_size, self.n_filter, self.stride),
            '(%d * %d * %d)' % (
                self.input_shape[1], self.input_shape[2], self.input_shape[3]),
            '(%d * %d * %d)' % (
                self.output_shape[1], self.output_shape[2], self.output_shape[3])))
        
    def get_output(self, input):
        with tf.name_scope('%s_cal' % (self.name)) as scope:
            # hidden states
            self.conv = tf.nn.conv2d(
                input=input, filter=self.weight, 
                strides=[1, self.stride, self.stride, 1], padding='SAME', name='cal_conv')
            
            # batch normalization 技术
            if self.batch_normal:
                mean, variance = tf.nn.moments(self.conv, axes=[0, 1, 2], keep_dims=False, name='get_moments')
                self.hidden = tf.nn.batch_normalization(
                    self.conv, mean, variance, self.bias, self.gamma, self.epsilon, name='cal_bn')
            else:
                self.hidden = self.conv + self.bias
                
            # activation
            if self.activation == 'relu':
                self.output = tf.nn.relu(self.hidden)
            elif self.activation == 'tanh':
                self.output = tf.nn.tanh(self.hidden)
            elif self.activation == 'leaky_relu':
                self.output = self.leaky_relu(self.hidden)
            elif self.activation == 'sigmoid':
                self.output = tf.nn.sigmoid(self.hidden)
            elif self.activation == 'none':
                self.output = self.hidden
            
            # gradient constraint
            g = tf.get_default_graph()
            with g.gradient_override_map({"Identity": "CustomClipGrad"}):
                self.output = tf.identity(self.output, name="Identity")
        
        return self.output
    
    def leaky_relu(self, input):
        output = tf.maximum(self.leaky_scale * input, input, name='leaky_relu')
        
        return output

    @tf.RegisterGradient("CustomClipGrad")
    def _clip_grad(unused_op, grad):
        return tf.clip_by_value(grad, -1, 1)

    def random_normal(self, shape, mean=0.0, stddev=1.0):
        epsilon = 1e-5
        twopi = 2.0 * math.pi

        n_dims = 1
        for dim in shape:
            n_dims *= dim
        array = numpy.zeros((n_dims, ), dtype='float32')
        
        for i in range(int(n_dims/2)):
            u1 = 0.0
            while u1 < epsilon:
                u1 = random.random()
                u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(twopi * u2)
            z1 = math.sqrt(-2.0 * math.log(u1)) * math.sin(twopi * u2)
            array[2*i] = z0 * stddev + mean
            array[2*i+1] = z1 * stddev + mean

        if n_dims % 2 == 1:
            while u1 < epsilon:
                u1 = random.random()
                u2 = random.random()
            z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(twopi * u2)
            array[n_dims-1] = z0

        array = numpy.reshape(array, shape)

        return array

    def rand_normal(self, shape, mean=0.0, stddev=1.0):
        import pdfinsight.ai.yolo_tf.src.tools.pyolo as pyolo
        n_dims = 1
        for dim in shape:
            n_dims *= dim
        array = numpy.zeros((n_dims, ), dtype='float32')
        
        for i in range(n_dims):
            array[i] = pyolo.rand_normal()
        
        array = numpy.reshape(array, shape)
        
        return array

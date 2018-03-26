# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import math
import tensorflow as tf
import random
import src.layer.utils as utils
from src.layer.batch_normal_layer import BatchNormalLayer


class ConvLayer:
    
    def __init__(self, y_size, x_size, y_stride, x_stride, n_filter, activation='relu',
                 batch_normal=False, weight_decay=None, name='conv',
                 input_shape=None, prev_layer=None):
        # params
        self.y_size = y_size
        self.x_size = x_size
        self.y_stride = y_stride
        self.x_stride = x_stride
        self.n_filter = n_filter
        self.activation = activation
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        self.name = name
        self.ltype = 'conv'
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.prev_layer = None
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 计算感受野
        self.feel_field = [1, 1]
        self.feel_field[0] = min(self.input_shape[0], 1 + int((self.y_size+1)/2))
        self.feel_field[1] = min(self.input_shape[1], 1 + int((self.x_size+1)/2))
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
        
        self.leaky_scale = tf.constant(0.1, dtype=tf.float32)
    
        with tf.name_scope('%s_def' % (self.name)) as scope:
            # 权重矩阵
            numpy.random.seed(0)
            scale = math.sqrt(2.0 / (self.y_size * self.x_size * self.input_shape[2]))
            init_value = scale * numpy.random.normal(size=[
                self.y_size, self.x_size, self.input_shape[2], self.n_filter], loc=0.0, scale=1.0)
            self.weight = tf.Variable(init_value, dtype=tf.float32, name='weight')
            
            # batch normalization 技术的参数
            if self.batch_normal:
                self.batch_normal_layer = BatchNormalLayer(self.n_filter, name=name)
            else:
                # 偏置向量
                self.bias = tf.Variable(
                    initial_value=tf.constant(0.0, shape=[self.n_filter]),
                    name='bias')
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [
            int(self.input_shape[0]/self.y_stride),
            int(self.input_shape[1]/self.x_stride), 
            self.n_filter]
        print('%-10s\t%-25s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '((%d, %d) / (%d, %d) * %d)' % (
                self.y_size, self.x_size, self.y_stride, self.x_stride, self.n_filter),
            '(%d, %d, %d)' % (
                self.input_shape[0], self.input_shape[1], self.input_shape[2]),
            '(%d, %d, %d)' % (
                self.output_shape[0], self.output_shape[1], self.output_shape[2]),
            '(%d, %d)' % (
                self.feel_field[0], self.feel_field[1])))
        self.calculation = self.output_shape[0] * self.output_shape[1] * \
            self.output_shape[2] * self.input_shape[2] * self.y_size * self.x_size
        
    def get_output(self, input, is_training=True):
        with tf.name_scope('%s_cal' % (self.name)) as scope:
            # hidden states
            self.conv = tf.nn.conv2d(
                input=input, filter=self.weight, 
                strides=[1, self.y_stride, self.x_stride, 1], padding='SAME', name='cal_conv')
            
            # batch normalization 技术
            if self.batch_normal:
                self.hidden = self.batch_normal_layer.get_output(self.conv, is_training=is_training)
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

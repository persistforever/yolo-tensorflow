# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf
from src.layer.batch_normal_layer import BatchNormalLayer


class DenseLayer:
    
    def __init__(self, hidden_dim, activation='relu', dropout=False, 
                 keep_prob=None, batch_normal=False, weight_decay=None, name='dense',
                 input_shape=None, prev_layer=None):
        # params
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        self.name = name
        self.ltype = 'dense'
        if prev_layer:
            self.prev_layer = prev_layer
            self.input_shape = prev_layer.output_shape
        elif input_shape:
            self.prev_layer = None
            self.input_shape = input_shape
        else:
            raise('ERROR: prev_layer or input_shape cannot be None!')
        
        # 权重矩阵
        self.weight = tf.Variable(
            initial_value=tf.random_normal(
                shape=[self.input_shape[0], self.hidden_dim],
                mean=0.0, stddev=numpy.sqrt(2.0 / self.input_shape[0])),
            name='W_%s' % (name))
            
        # batch normalization 技术的参数
        if self.batch_normal:
            self.batch_normal_layer = BatchNormalLayer(self.hidden_dim, name=name)
        else:
            # 偏置向量
            self.bias = tf.Variable(
                initial_value=tf.constant(0.0, shape=[self.hidden_dim]),
                name='bias')
        
        # dropout 技术
        if self.dropout:
            self.keep_prob = keep_prob
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.hidden_dim]
        print('%-10s\t%-25s\t%-20s\t%s' % (
            self.name, 
            '(%d)' % (self.hidden_dim),
            '(%d)' % (self.input_shape[0]),
            '(%d)' % (self.output_shape[0])))
        self.calculation = self.output_shape[0] * self.input_shape[0]
        
    def get_output(self, input):
        
        # hidden states
        intermediate = tf.matmul(input, self.weight)
        
        # batch normalization 技术
        if self.batch_normal:
            if self.batch_normal:
                self.hidden = self.batch_normal_layer.get_output(self.conv, is_training=is_training)
            else:
                self.hidden = self.conv + self.bias
        else:
            self.hidden = intermediate + self.bias
            
        # dropout 技术
        if self.dropout:
            self.hidden = tf.nn.dropout(self.hidden, keep_prob=self.keep_prob)
            
        # activation
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.hidden)
        elif self.activation == 'tanh':
            self.output = tf.nn.tanh(self.hidden)
        elif self.activation == 'softmax':
            self.output = tf.nn.softmax(self.hidden)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(self.hidden)
        elif self.activation == 'leaky_relu':
            self.output = self.leaky_relu(self.hidden)
        elif self.activation == 'none':
            self.output = self.hidden
        
        # gradient constraint
        g = tf.get_default_graph()
        with g.gradient_override_map({"Identity": "CustomClipGrad"}):
            self.output = tf.identity(self.output, name="Identity")
        
        return self.output
    
    def leaky_relu(self, input):
        hidden = tf.cast(input, dtype=tf.float32)
        mask = tf.cast((hidden > 0), dtype=tf.float32)
        output = 1.0 * mask * hidden + 0.1 * (1 - mask) * hidden
        
        return output

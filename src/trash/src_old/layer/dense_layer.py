# -*- coding: utf8 -*-
# author: ronniecao
import numpy
import tensorflow as tf


class DenseLayer:
    
    def __init__(self, input_shape, hidden_dim, activation='relu', dropout=False, 
                 keep_prob=None, batch_normal=False, weight_decay=None, name='dense'):
        # params
        self.input_shape = input_shape
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout = dropout
        self.batch_normal = batch_normal
        self.weight_decay = weight_decay
        self.name = name
        
        # 权重矩阵
        self.weight = tf.Variable(
            initial_value=tf.random_normal(
                shape=[self.input_shape[1], self.hidden_dim],
                mean=0.0, stddev=numpy.sqrt(2.0 / self.input_shape[1])),
            name='W_%s' % (name))
        
        # weight decay技术
        if weight_decay:
            weight_decay = tf.multiply(tf.nn.l2_loss(self.weight), self.weight_decay)
            tf.add_to_collection('losses', weight_decay)
            
        # 偏置向量
        self.bias = tf.Variable(
            initial_value=tf.constant(
                0.0, shape=[self.hidden_dim]),
            name='b_%s' % (name))
        
        # batch normalization 技术的参数
        if self.batch_normal:
            self.epsilon = 1e-5
            self.gamma = tf.Variable(
                initial_value=tf.constant(
                    1.0, shape=[self.hidden_dim]),
            name='gamma_%s' % (name))
            
        # dropout 技术
        if self.dropout:
            self.keep_prob = keep_prob
        
        # 打印网络权重、输入、输出信息
        # calculate input_shape and output_shape
        self.output_shape = [self.input_shape[0], self.hidden_dim]
        print('%-10s\t%-20s\t%-20s\t%s' % (
            self.name, 
            '(%d)' % (self.hidden_dim),
            '(%d)' % (self.input_shape[1]),
            '(%d)' % (self.output_shape[1])))
        
    def get_output(self, input):
        
        # hidden states
        intermediate = tf.matmul(input, self.weight)
        
        # batch normalization 技术
        if self.batch_normal:
            mean, variance = tf.nn.moments(intermediate, axes=[0])
            self.hidden = tf.nn.batch_normalization(
                intermediate, mean, variance, self.bias, self.gamma, self.epsilon)
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

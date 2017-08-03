# -*- coding: utf8 -*-
# author: ronniecao
import sys
import os
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer

class TinyYolo():
    
    def __init__(self, n_channel=3, n_classes=1, image_size=288, max_objects_per_image=20,
                 cell_size=7, box_per_cell=5):
        # 设置参数
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_channel = n_channel
        self.max_objects = max_objects_per_image
        self.cell_size = cell_size
        self.n_boxes = box_per_cell
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.image_size, self.image_size, self.n_channel], name='images')
        self.labels = tf.placeholder(
            dtype=tf.float32, shape=[
                None, self.max_objects, 5], name='labels')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        
    def inference(self, images):
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(None, self.image_size, self.image_size, self.n_channel), 
            n_size=3, n_filter=16, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')
        
        conv_layer2 = ConvLayer(
            input_shape=(None, int(self.image_size/2), int(self.image_size/2), 16), 
            n_size=3, n_filter=32, stride=1, activation='relu',
            batch_normal=False, weight_decay=None, name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        
        conv_layer3 = ConvLayer(
            input_shape=(None, int(self.image_size/4), int(self.image_size/4), 32),
            n_size=3, n_filter=64, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        
        conv_layer4 = ConvLayer(
            input_shape=(None, int(self.image_size/8), int(self.image_size/8), 64),
            n_size=3, n_filter=128, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv4')
        pool_layer4 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool4')
        
        conv_layer5 = ConvLayer(
            input_shape=(None, int(self.image_size/16), int(self.image_size/16), 128),
            n_size=3, n_filter=256, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv5')
        pool_layer5 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool5')
        
        conv_layer6 = ConvLayer(
            input_shape=(None, int(self.image_size/32), int(self.image_size/32), 256),
            n_size=3, n_filter=512, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv6')
        pool_layer6 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool6')
        
        conv_layer7 = ConvLayer(
            input_shape=(None, int(self.image_size/32), int(self.image_size/32), 512),
            n_size=3, n_filter=1024, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(None, int(self.image_size/32), int(self.image_size/32), 1024),
            n_size=3, n_filter=1024, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv8')
        
        dense_layer1 = DenseLayer(
            input_shape=(None, int(self.image_size/32) * int(self.image_size/32) * 1024), 
            hidden_dim=self.cell_size * self.cell_size * (self.n_classes + self.n_boxes * 5), 
            activation='none', dropout=False, keep_prob=None, 
            batch_normal=False, weight_decay=None, name='dense1')
        
        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=self.images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv3)
        hidden_conv4 = conv_layer4.get_output(input=hidden_pool3)
        hidden_pool4 = pool_layer4.get_output(input=hidden_conv4)
        hidden_conv5 = conv_layer5.get_output(input=hidden_pool4)
        hidden_pool5 = pool_layer5.get_output(input=hidden_conv5)
        hidden_conv6 = conv_layer6.get_output(input=hidden_pool5)
        hidden_pool6 = pool_layer6.get_output(input=hidden_conv6)
        hidden_conv7 = conv_layer7.get_output(input=hidden_pool6)
        hidden_conv8 = conv_layer8.get_output(input=hidden_conv7)
        input_dense1 = tf.reshape(hidden_conv8, shape=[
            -1, int(self.image_size/32) * int(self.image_size/32) * 1024])
        output = dense_layer1.get_output(input=input_dense1)
        
        # 网络输出
        return output
        
    def train(self, dataloader, backup_path, n_epoch=5, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=1000)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        # 模型训练
        for epoch in range(0, n_epoch+1):
            # 数据增强
            train_images = dataloader.data_augmentation(dataloader.train_images, mode='train',
                flip=True, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            train_labels = dataloader.train_labels
            valid_images = dataloader.data_augmentation(dataloader.valid_images, mode='test',
                flip=False, crop=True, crop_shape=(24,24,3), whiten=True, noise=False)
            valid_labels = dataloader.valid_labels
            # 开始本轮的训练
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [_, avg_loss] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 0.5})
            # 在训练之后，获得本轮的训练集损失值和准确率
            train_accuracy, train_loss = 0.0, 0.0
            for i in range(0, dataloader.n_train, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_labels = train_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                train_accuracy += avg_accuracy * batch_images.shape[0]
                train_loss += avg_loss * batch_images.shape[0]
            train_accuracy = 1.0 * train_accuracy / dataloader.n_train
            train_loss = 1.0 * train_loss / dataloader.n_train
            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_accuracy, valid_loss = 0.0, 0.0
            for i in range(0, dataloader.n_valid, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_labels = valid_labels[i: i+batch_size]
                [avg_accuracy, avg_loss] = self.sess.run(
                    fetches=[self.accuracy, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.labels: batch_labels, 
                               self.keep_prob: 1.0})
                valid_accuracy += avg_accuracy * batch_images.shape[0]
                valid_loss += avg_loss * batch_images.shape[0]
            valid_accuracy = 1.0 * valid_accuracy / dataloader.n_valid
            valid_loss = 1.0 * valid_loss / dataloader.n_valid
            print('epoch: %d, train precision: %.6f, train loss: %.6f, valid precision: %.6f, valid loss: %.6f' % (
                epoch, train_accuracy, train_loss, valid_accuracy, valid_loss))
            sys.stdout.flush()
            # 保存模型
            saver_path = self.saver.save(
                self.sess, os.path.join(backup_path, 'model.ckpt'))
            if epoch <= 100 and epoch % 10 == 0 or epoch <= 1000 and epoch % 100 == 0 or \
                epoch <= 10000 and epoch % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
                
    def test(self, dataloader, backup_path, epoch, batch_size=128):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        # 在测试集上计算准确率
        accuracy_list = []
        test_images = dataloader.data_augmentation(dataloader.test_images,
            flip=True, crop=True, shape=(24,24,3), whiten=True, noise=False)
        test_labels = dataloader.test_labels
        for i in range(0, dataloader.n_test, batch_size):
            batch_images = test_images[i: i+batch_size]
            batch_labels = test_labels[i: i+batch_size]
            [avg_accuracy] = self.sess.run(
                fetches=[self.accuracy], 
                feed_dict={self.images:batch_images, 
                           self.labels:batch_labels,
                           self.keep_prob:1.0})
            accuracy_list.append(avg_accuracy)
        print('test precision: %.4f' % (numpy.mean(accuracy_list)))
            
    def debug(self):
        self.observe = self.inference(self.images)
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(tf.global_variables_initializer())
        # 运行
        [temp] = self.sess.run(
            fetches=[self.observe],
            feed_dict={self.images: numpy.random.random(size=[128, 384, 384, 3]),
                       self.labels: numpy.random.randint(low=0, high=1, size=[128, 20, 5]),
                       self.keep_prob: 1.0})
        print(temp.shape)
# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import math
import numpy
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer


class TinyYolo():
    
    def __init__(self, n_channel=3, n_classes=1, image_size=288, max_objects_per_image=20,
                 cell_size=7, box_per_cell=5, object_scala=1, nobject_scala=1,
                 coord_scala=1, class_scala=1, batch_size=2, nobject_thresh=0.6,
                 recall_thresh=0.5):
        # 设置参数
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_channel = n_channel
        self.max_objects = max_objects_per_image
        self.cell_size = cell_size
        self.n_boxes = box_per_cell
        self.class_scala = float(class_scala)
        self.object_scala = float(object_scala)
        self.nobject_scala = float(nobject_scala)
        self.coord_scala = float(coord_scala)
        self.batch_size = batch_size
        self.nobject_thresh = nobject_thresh
        self.recall_thresh = recall_thresh
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.image_size, self.image_size, self.n_channel], 
            name='images')
        self.class_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.n_classes], 
            name='class_labels')
        self.class_masks = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size],
            name='class_masks')
        self.box_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.max_objects, 6], 
            name='box_labels')
        self.object_nums = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, ],
            name='object_nums')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        self.global_step = tf.Variable(
            0, dtype=tf.int32, name='global_step')
        
        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.class_loss, self.coord_loss, self.object_loss, self.nobject_loss, \
            self.iou_value, self.object_value, self.nobject_value, self.recall_value = \
            self.loss(self.logits)
            
        # 目标函数和优化器
        tf.add_to_collection('losses', self.class_loss)
        tf.add_to_collection('losses', self.coord_loss)
        tf.add_to_collection('losses', self.object_loss)
        tf.add_to_collection('losses', self.nobject_loss)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        
        # 设置学习率
        lr = tf.cond(tf.less(self.global_step, 100), 
                     lambda: tf.constant(0.001),
                     lambda: tf.cond(tf.less(self.global_step, 8000),
                                     lambda: tf.constant(0.01),
                                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                                     lambda: tf.constant(0.001),
                                                     lambda: tf.constant(0.0001))))
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr, momentum=0.9).minimize(
                self.avg_loss, global_step=self.global_step)
        
    def inference(self, images):
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(self.batch_size, self.image_size, self.image_size, self.n_channel), 
            n_size=7, n_filter=64, stride=2, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv1')
        pool_layer1 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/2), 
                         int(self.image_size/2), 64),
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool1')
        
        conv_layer2 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 64), 
            n_size=3, n_filter=192, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv2')
        pool_layer2 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 192),
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool2')
        
        conv_layer3 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 192),
            n_size=1, n_filter=128, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv3')
        conv_layer4 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 128),
            n_size=3, n_filter=256, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv4')
        conv_layer5 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 256),
            n_size=1, n_filter=128, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv5')
        conv_layer6 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 128),
            n_size=3, n_filter=256, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv6')
        pool_layer3 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 256),
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool3')
        
        conv_layer7 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 256),
            n_size=1, n_filter=256, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 256),
            n_size=3, n_filter=512, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv8')
        conv_layer9 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 512),
            n_size=1, n_filter=512, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv9')
        conv_layer10 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 512),
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv10')
        pool_layer4 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 1024),
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool4')
        
        conv_layer11 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 1024),
            n_size=1, n_filter=512, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv11')
        conv_layer12 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 512),
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv12')
        conv_layer13 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 1024),
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv13')
        
        dense_layer1 = DenseLayer(
            input_shape=(self.batch_size, 
                         int(self.image_size/32) * int(self.image_size/32) * 1024), 
            hidden_dim=4096, 
            activation='leaky_relu', dropout=True, keep_prob=self.keep_prob,
            batch_normal=True, weight_decay=5e-4, name='dense1')
        
        dense_layer2 = DenseLayer(
            input_shape=(self.batch_size, 4096), 
            hidden_dim=self.cell_size * self.cell_size * (self.n_classes + self.n_boxes * 5), 
            activation='sigmoid', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=5e-4, name='dense2')
        
        # 数据流
        print('\n%-10s\t%-20s\t%-20s\t%s' % (
            'Name', 'Filter', 'Input', 'Output'))
        hidden_conv1 = conv_layer1.get_output(input=images)
        hidden_pool1 = pool_layer1.get_output(input=hidden_conv1)
        
        hidden_conv2 = conv_layer2.get_output(input=hidden_pool1)
        hidden_pool2 = pool_layer2.get_output(input=hidden_conv2)
        
        hidden_conv3 = conv_layer3.get_output(input=hidden_pool2)
        hidden_conv4 = conv_layer4.get_output(input=hidden_conv3)
        hidden_conv5 = conv_layer5.get_output(input=hidden_conv4)
        hidden_conv6 = conv_layer6.get_output(input=hidden_conv5)
        hidden_pool3 = pool_layer3.get_output(input=hidden_conv6)
        
        hidden_conv7 = conv_layer7.get_output(input=hidden_pool3)
        hidden_conv8 = conv_layer8.get_output(input=hidden_conv7)
        hidden_conv9 = conv_layer9.get_output(input=hidden_conv8)
        hidden_conv10 = conv_layer10.get_output(input=hidden_conv9)
        hidden_pool4 = pool_layer4.get_output(input=hidden_conv10)
        
        hidden_conv11 = conv_layer11.get_output(input=hidden_pool4)
        hidden_conv12 = conv_layer12.get_output(input=hidden_conv11)
        hidden_conv13 = conv_layer13.get_output(input=hidden_conv12)
        
        input_dense1 = tf.reshape(hidden_conv13, shape=[
            -1, int(self.image_size/32) * int(self.image_size/32) * 1024])
        hidden_dense1 = dense_layer1.get_output(input=input_dense1)
        logits = dense_layer2.get_output(input=hidden_dense1)
        
        print()
        sys.stdout.flush()
        # 网络输出
        return logits
    
    def _loss_one_example_cond(self, num, object_num, batch, coord_loss, object_loss, 
                              nobject_loss, iou_value, object_value, recall_value):
        
        return num < object_num
    
    def _loss_one_example_body(self, num, object_num, batch, coord_loss, object_loss, 
                              nobject_loss, iou_value, object_value, recall_value):
        # 构造box_label和object_mask
        # 如果cell中有物体，object_mask则为1，如果cell中没有物体，则为0
        # 如果cell中有物体，box_label的第一个box则为四个坐标，其他box为0，如果cell中没有物体，则为0
        cell_x = self.box_labels[batch, num, 0]
        cell_y = self.box_labels[batch, num, 1]
        object_mask = tf.ones(
            shape=(1, 1), dtype=tf.float32)
        padding = tf.cast([[cell_y, self.cell_size-cell_y-1], 
                           [cell_x, self.cell_size-cell_x-1]], dtype=tf.int32)
        object_mask = tf.reshape(
            tf.pad(object_mask, paddings=padding, mode='CONSTANT'),
            shape=(self.cell_size, self.cell_size, 1, 1))
        
        box_label = tf.cast(self.box_labels[batch,num,2:6], dtype=tf.float32)
        box_label = tf.reshape(box_label, shape=(1, 1, 1, 4))
        padding = tf.cast([[cell_y, self.cell_size-cell_y-1], 
                           [cell_x, self.cell_size-cell_x-1],
                           [0, self.n_boxes-1], [0, 0]], dtype=tf.int32)
        box_label = tf.pad(box_label, paddings=padding, mode='CONSTANT')
        
        # 计算iou_matrix，表示每个cell中，每个box与这个cell中真实物体的iou值
        iou_matrix = self.iou(self.box_preds[batch,:,:,:,0:4], box_label)
        iou_matrix_max = tf.reduce_max(iou_matrix, 2, keep_dims=True)
        iou_matrix_mask = tf.cast(
            (iou_matrix >= iou_matrix_max), dtype=tf.float32) * object_mask
            
        # 计算nobject_loss
        # nobject_pred为box_pred的值，尺寸为(cell_size, cell_size, n_box, 1)
        # 每一个cell中，有object，并且iou > nobject_thresh，则不计算，否则为0
        nobject_mask = tf.ones_like(iou_matrix_mask) - iou_matrix_mask
        nobject_label = tf.zeros(
            shape=(self.cell_size, self.cell_size, self.n_boxes, 1),
            dtype=tf.float32)
        nobject_pred = self.box_preds[batch,:,:,:,4:]
        nobject_loss += tf.nn.l2_loss(
            (nobject_pred - nobject_label) * nobject_mask)
        
        # 计算object_loss
        # object_pred为box_pred的值，尺寸为(cell_size, cell_size, n_box, 1)
        # 每一个cell中，有object，并且iou最大的那个box的object_label为iou，其余为0，
        # object_label尺寸为(cell_size, cell_size, n_box, 1)
        object_label = iou_matrix
        object_pred = self.box_preds[batch,:,:,:,4:]
        object_loss += tf.nn.l2_loss(
            (object_pred - object_label) * iou_matrix_mask)
        
        # 计算coord_loss
        # coord_pred为box_pred的值，尺寸为(cell_size, cell_size, n_box, 1)
        # 每一个cell中，有object，并且iou最大的那个box的coord_label为真实的label，其余为0，
        # coord_label尺寸为(cell_size, cell_size, n_box, 1)
        coord_label = box_label
        coord_pred = self.box_preds[batch,:,:,:,0:4]
        coord_loss += tf.nn.l2_loss(
            (coord_pred[:,:,:,0:2] - coord_label[:,:,:,0:2]) * iou_matrix_mask)
        coord_loss += tf.nn.l2_loss(
            (tf.sqrt(coord_pred[:,:,:,2:4]) - tf.sqrt(coord_label[:,:,:,2:4])) * \
            iou_matrix_mask)
        
        # 计算iou_value
        # 每一个cell中，有object，并且iou最大的那个对应的iou
        iou_value += tf.reduce_sum(
            iou_matrix * iou_matrix_mask, axis=[0,1,2,3])
        
        # 计算object_value
        # 每一个cell中，有object，并且iou最大的那个对应的box_pred中的confidence
        object_value += tf.reduce_sum(
            self.box_preds[batch,:,:,:,4:] * iou_matrix_mask, axis=[0,1,2,3])
            
        # 计算recall_value
        # 每一个cell中，有object，并且iou最大的哪个对应的iou如果大于recall_thresh，则加1
        recall_mask = tf.cast(
            (iou_matrix * iou_matrix_mask > self.recall_thresh), dtype=tf.float32)
        recall_value += tf.reduce_sum(
                recall_mask, axis=[0,1,2,3])
        num += 1
        
        return num, object_num, batch, coord_loss, object_loss, \
            nobject_loss, iou_value, object_value, recall_value
    
    def loss(self, logits):
        logits = tf.reshape(
            logits, shape=[self.batch_size, self.cell_size, self.cell_size, 
                           self.n_classes + self.n_boxes * 5])
        
        # 获取class_pred和box_pred
        class_preds = logits[:,:,:,0:self.n_classes]
        self.box_preds = tf.reshape(
            logits[:,:,:,self.n_classes:], 
            shape=[self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 5])
        
        class_loss = 0.0
        coord_loss = 0.0
        object_loss = 0.0
        nobject_loss = 0.0
        iou_value = 0.0
        object_value = 0.0
        nobject_value = 0.0
        recall_value = 0.0
        
        for i in range(self.batch_size):
            
            # 计算class_loss
            class_pred = class_preds[i,:,:,:]
            class_label = self.class_labels[i,:,:,:]
            class_mask = tf.reshape(
                self.class_masks[i,:,:], 
                shape=[self.cell_size, self.cell_size, 1])
            class_loss += tf.nn.l2_loss(
                (class_pred - class_label) * class_mask) / \
                (self.cell_size * self.cell_size * 1.0)
                
            # 循环计算每一个example
            results = tf.while_loop(
                cond=self._loss_one_example_cond, 
                body=self._loss_one_example_body, 
                loop_vars=[tf.constant(0), self.object_nums[i], i,
                           tf.constant(0.0), tf.constant(0.0), tf.constant(0.0),
                           tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)])
            coord_loss += results[3]
            object_loss += results[4]
            nobject_loss += results[5]
            iou_value += results[6]
            object_value += results[7]
            recall_value += results[8]
            
            # 计算nobject_value
            # 所有的box_pred中的confidence
            nobject_value += tf.reduce_sum(
                self.box_preds[i,:,:,:,4:], axis=[0,1,2,3])
            
        # 目标函数值
        class_loss = class_loss * self.class_scala / self.batch_size
        coord_loss = coord_loss * self.coord_scala / self.batch_size
        object_loss = object_loss * self.object_scala / self.batch_size
        nobject_loss = nobject_loss * self.nobject_scala / self.batch_size
        # 观察值
        iou_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        object_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        nobject_value /= (self.cell_size * self.cell_size * self.n_boxes * self.batch_size)
        recall_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        
        return class_loss, coord_loss, object_loss, nobject_loss, \
            iou_value, object_value, nobject_value, recall_value
              
    def iou(self, box_pred, box_label):
        box1 = tf.stack([
            box_pred[:,:,:,0] - box_pred[:,:,:,2] / 2,
            box_pred[:,:,:,1] - box_pred[:,:,:,3] / 2,
            box_pred[:,:,:,0] + box_pred[:,:,:,2] / 2,
            box_pred[:,:,:,1] + box_pred[:,:,:,3] / 2])
        box1 = tf.transpose(box1, perm=[1, 2, 3, 0])
        box2 = tf.stack([
            box_label[:,:,:,0] - box_label[:,:,:,2] / 2,
            box_label[:,:,:,1] - box_label[:,:,:,3] / 2,
            box_label[:,:,:,0] + box_label[:,:,:,2] / 2,
            box_label[:,:,:,1] + box_label[:,:,:,3] / 2])
        box2 = tf.transpose(box2, perm=[1, 2, 3, 0])
        
        left_top = tf.maximum(box1[:,:,:,0:2], box2[:,:,:,0:2])
        right_bottom = tf.minimum(box1[:,:,:,2:4], box2[:,:,:,2:4])
        intersection = right_bottom - left_top
        inter_area = intersection[:,:,:,0] * intersection[:,:,:,1]
        mask = tf.cast(intersection[:,:,:,0] > 0, tf.float32) * \
            tf.cast(intersection[:,:,:,1] > 0, tf.float32)
        inter_area = inter_area * mask
        box1_area = (box1[:,:,:,2] - box1[:,:,:,0]) * (box1[:,:,:,3] - box1[:,:,:,1])
        box2_area = (box2[:,:,:,2] - box2[:,:,:,0]) * (box2[:,:,:,3] - box2[:,:,:,1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        return tf.reshape(iou, shape=[self.cell_size, self.cell_size, self.n_boxes, 1])
        
    def train(self, processor, backup_path, n_iters=500000, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=1)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
                
        # 模型训练
        process_images = 0
        train_avg_loss, train_class_loss, train_coord_loss, \
            train_object_loss, train_nobject_loss = 0.0, 0.0, 0.0, 0.0, 0.0
        train_iou_value, train_object_value, \
            train_nobject_value, train_recall_value = 0.0, 0.0, 0.0, 0.0 
        
        for n_iter in range(1, n_iters+1):
            # 训练一个batch，计算从准备数据到训练结束的时间
            start_time = time.time()
            
            # 获取数据并进行数据增强
            batch_images, batch_labels = processor.get_train_batch(batch_size)
            batch_images, batch_labels = processor.data_augmentation(
                batch_images, batch_labels, mode='train',
                flip=True, whiten=True, resize=True, jitter=0.2)
            batch_class_labels, batch_class_masks, batch_box_labels, batch_object_nums = \
                processor.process_batch_labels(batch_labels)
            
            [_, avg_loss, class_loss, coord_loss, object_loss, nobject_loss,
             iou_value, object_value, nobject_value, recall_value] = self.sess.run(
                fetches=[self.optimizer, self.avg_loss,
                         self.class_loss, self.coord_loss,
                         self.object_loss, self.nobject_loss,
                         self.iou_value, self.object_value,
                         self.nobject_value, self.recall_value], 
                feed_dict={self.images: batch_images, 
                           self.class_labels: batch_class_labels, 
                           self.class_masks: batch_class_masks,
                           self.box_labels: batch_box_labels,
                           self.object_nums: batch_object_nums,
                           self.keep_prob: 0.5})
                
            end_time = time.time()
            
            train_avg_loss += avg_loss
            train_class_loss += class_loss
            train_coord_loss += coord_loss
            train_object_loss += object_loss
            train_nobject_loss += nobject_loss
            train_iou_value += iou_value
            train_object_value += object_value
            train_nobject_value += nobject_value
            train_recall_value += recall_value
            
            process_images += batch_size
            speed = 1.0 * batch_size / (end_time - start_time)
                
            # 每1轮训练观测一次train_loss    
            print('{TRAIN} iter[%d], train loss: %.6f, class_loss: %.6f, coord_loss: %.6f, '
                  'object_loss: %.6f, nobject_loss: %.6f, image_nums: %d, '
                  'speed: %.2f images/s' % (
                n_iter, train_avg_loss, train_class_loss, train_coord_loss, 
                train_object_loss, train_nobject_loss, process_images, speed))
            sys.stdout.flush()
            
            train_avg_loss, train_class_loss, train_coord_loss, \
                train_object_loss, train_nobject_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 每1轮观测一次训练集evaluation
            print('{TRAIN} iter[%d], iou: %.6f, object: %.6f, '
                  'nobject: %.6f, recall: %.6f' % (
                n_iter, train_iou_value, train_object_value, 
                train_nobject_value, train_recall_value))
            sys.stdout.flush()
            
            train_iou_value, train_object_value, \
                train_nobject_value, train_recall_value = 0.0, 0.0, 0.0, 0.0 
            
            # 每100轮观测一次验证集evaluation
            if n_iter % 100 == 0:
                valid_iou_value, valid_object_value, \
                    valid_nobject_value, valid_recall_value = 0.0, 0.0, 0.0, 0.0 
                
                for i in range(0, processor.n_valid-batch_size, batch_size):
                    
                    # 获取数据并进行数据增强
                    batch_images, batch_labels = processor.get_valid_batch(i, batch_size)
                    batch_images, batch_labels = processor.data_augmentation(
                        batch_images, batch_labels, mode='test',
                        flip=False,
                        whiten=True,
                        resize=True)
                    batch_class_labels, batch_class_masks, batch_box_labels, batch_object_nums = \
                        processor.process_batch_labels(batch_labels)
                    
                    [iou_value, object_value,
                     nobject_value, recall_value] = self.sess.run(
                        fetches=[self.iou_value, self.object_value,
                                 self.nobject_value, self.recall_value],
                        feed_dict={self.images: batch_images, 
                                   self.class_labels: batch_class_labels, 
                                   self.class_masks: batch_class_masks,
                                   self.box_labels: batch_box_labels,
                                   self.object_nums: batch_object_nums,
                                   self.keep_prob: 1.0})
                     
                    valid_iou_value += iou_value * batch_size
                    valid_object_value += object_value * batch_size
                    valid_nobject_value += nobject_value * batch_size
                    valid_recall_value += recall_value * batch_size
                    
                valid_iou_value /= i
                valid_object_value /= i
                valid_nobject_value /= i
                valid_recall_value /= i
                
                print('{VALID} iter[%d], valid: iou: %.8f, object: %.8f, '
                      'nobject: %.8f, recall: %.8f' % (
                    n_iter, valid_iou_value, valid_object_value, 
                    valid_nobject_value, valid_recall_value))
                sys.stdout.flush()
            
            # 每1000轮保存一次模型
            if n_iter % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model.ckpt'))
                
        self.sess.close()
                
    def test(self, processor, backup_path, n_iter=0, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_path, 'model_%d.ckpt' % (epoch))
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        
        # 在测试集上计算
        for i in range(0, processor.n_test-batch_size, batch_size):
            batch_images, batch_class_labels, batch_class_masks, batch_box_labels, \
                batch_object_masks, batch_nobject_masks, batch_object_nums = \
                processor.get_test_batch(i, batch_size)
                
            [logits] = self.sess.run(
                fetches=[self.logits],
                feed_dict={self.images: batch_images, 
                           self.class_labels: batch_class_labels, 
                           self.class_masks: batch_class_masks,
                           self.box_labels: batch_box_labels,
                           self.object_masks: batch_object_masks,
                           self.nobject_masks: batch_nobject_masks,
                           self.object_num: batch_object_nums,
                           self.keep_prob: 1.0})
            
            logits = tf.reshape(
                logits, shape=[self.batch_size, self.cell_size, self.cell_size, 
                               self.n_classes + self.n_boxes * 5])
            class_preds = logits[:,:,:,0:self.n_classes]
            box_preds = tf.reshape(
                logits[:,:,:,self.n_classes:], 
                shape=[self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 5])
        
            for j in range(batch_images.shape[0]):
                image = batch_images[j]
                # 画真实的框
                for x in range(self.cell_size):
                    for y in range(self.cell_size):
                        for n in range(self.max_objects):
                            box = batch_box_labels[i, x, y, n]
                            if box[4] == 1.0:
                                xmin = int((box[0] - box[2] / 2.0) * image.shape[0])
                                xmax = int((box[0] + box[2] / 2.0) * image.shape[0])
                                ymin = int((box[1] - box[3] / 2.0) * image.shape[1])
                                ymax = int((box[1] + box[3] / 2.0) * image.shape[1])
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 99, 71))
                # 画预测的框
                for x in range(self.cell_size):
                    for y in range(self.cell_size):
                        class_pred = numpy.argmax(class_preds[j, x, y])
                        for n in range(self.n_boxes):
                            box = box_preds[j, x, y]
                            if box[4] >= 0.25:
                                xmin = int((box[0] - box[2] / 2.0) * image.shape[0])
                                xmax = int((box[0] + box[2] / 2.0) * image.shape[0])
                                ymin = int((box[1] - box[3] / 2.0) * image.shape[1])
                                ymax = int((box[1] + box[3] / 2.0) * image.shape[1])
                                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (238, 130, 238))
                                
                plt.imshow(batch_images[0])
                plt.show()
            exit()
            
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
        self.sess.close()
            
    def debug(self, processor):
        # 处理数据
        train_class_labels, train_object_masks, train_nobject_masks, \
            train_box_labels, train_box_masks = self.process_labels_cpu(processor.train_labels)
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
        self.sess.close()
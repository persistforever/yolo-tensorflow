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
                self.batch_size, self.cell_size, self.cell_size, self.max_objects, 4], 
            name='box_labels')
        self.object_masks = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects],
            name='object_masks')
        self.object_nums = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, ],
            name='object_nums')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        
        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.class_loss, self.coord_loss, self.object_loss, self.nobject_loss, \
            self.iou_value, self.object_value, self.nobject_value, self.recall_value = \
            self.loss(self.logits)
        # 目标函数和优化器
        # tf.add_to_collection('losses', self.class_loss)
        tf.add_to_collection('losses', self.coord_loss)
        # tf.add_to_collection('losses', self.object_loss)
        # tf.add_to_collection('losses', self.nobject_loss)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.avg_loss)
        
    def inference(self, images):
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(self.batch_size, self.image_size, self.image_size, self.n_channel), 
            n_size=3, n_filter=16, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool1')
        
        conv_layer2 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/2), int(self.image_size/2), 16), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool2')
        
        conv_layer3 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 32),
            n_size=3, n_filter=64, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool3')
        
        conv_layer4 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 64),
            n_size=3, n_filter=128, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv4')
        pool_layer4 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool4')
        
        conv_layer5 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 128),
            n_size=3, n_filter=256, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv5')
        pool_layer5 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=True, name='pool5')
        
        conv_layer6 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 256),
            n_size=3, n_filter=512, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv6')
        conv_layer7 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 512),
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 1024),
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv8')
        
        dense_layer1 = DenseLayer(
            input_shape=(self.batch_size, int(self.image_size/32) * int(self.image_size/32) * 1024), 
            hidden_dim=self.cell_size * self.cell_size * (self.n_classes + self.n_boxes * 5), 
            activation='sigmoid', dropout=False, keep_prob=None,
            batch_normal=False, weight_decay=5e-4, name='dense1')
        
        # 数据流
        hidden_conv1 = conv_layer1.get_output(input=images)
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
        hidden_conv7 = conv_layer7.get_output(input=hidden_conv6)
        hidden_conv8 = conv_layer8.get_output(input=hidden_conv7)
        input_dense1 = tf.reshape(hidden_conv8, shape=[
            -1, int(self.image_size/32) * int(self.image_size/32) * 1024])
        output = dense_layer1.get_output(input=input_dense1)
        
        # 网络输出
        return output
    
    def loss_one_example_cond(self, num, object_num, batch, coord_loss, object_loss, 
                              nobject_loss, iou_value, object_value, recall_value):
        
        return num < object_num
    
    def loss_one_example_body(self, num, object_num, batch, coord_loss, object_loss, 
                              nobject_loss, iou_value, object_value, recall_value):
        # 获取object_mask，表示如果cell中有物体，则为1，如果cell中没有物体，则为0
        object_mask = tf.reshape(
            self.object_masks[batch,:,:,num:num+1],
            shape=[self.cell_size, self.cell_size, 1, 1])
        
        # 计算iou_matrix，表示每个cell中，每个box与这个cell中真实物体的iou值
        iou_matrix = self.iou(self.box_preds[batch,:,:,:,0:4], 
                              self.box_labels[batch,:,:,num:num+1,0:4])
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
        coord_label = self.box_labels[batch,:,:,num:num+1,0:4]
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
                cond=self.loss_one_example_cond, 
                body=self.loss_one_example_body, 
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
        nobject_value /= self.batch_size
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
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, 
            max_to_keep=5)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
                
        # 模型训练
        process_images, train_loss = 0, 0.0
        class_loss, coord_loss, object_loss, nobject_loss = 0.0, 0.0, 0.0, 0.0
        for n_iter in range(1, n_iters+1):
            # 训练一个batch
            start_time = time.time()
            
            batch_images, batch_class_labels, batch_class_masks, batch_box_labels, \
                batch_object_masks, _, batch_object_nums = \
                processor.get_train_batch(batch_size)
            # 数据增强
            batch_images = processor.data_augmentation(
                batch_images, flip=False, 
                crop=True, padding=20, whiten=False)
            print(batch_images.shape)
            [_, avg_loss, class_l, coord_l, object_l, nobject_l] = self.sess.run(
                fetches=[self.optimizer, self.avg_loss,
                         self.class_loss, self.coord_loss, self.object_loss, self.nobject_loss], 
                feed_dict={self.images: batch_images, 
                           self.class_labels: batch_class_labels, 
                           self.class_masks: batch_class_masks,
                           self.box_labels: batch_box_labels,
                           self.object_masks: batch_object_masks,
                           self.object_nums: batch_object_nums,
                           self.keep_prob: 0.5})
                
            end_time = time.time()
            
            train_loss += avg_loss
            class_loss += class_l
            coord_loss += coord_l
            object_loss += object_l
            nobject_loss += nobject_l
            
            process_images += batch_images.shape[0]
            speed = 1.0 * batch_images.shape[0] / (end_time - start_time)
                
            # 每10轮训练观测一次train_loss
            if n_iter % 10 == 0:
                train_loss /= 10
                class_loss /= 10
                coord_loss /= 10
                object_loss /= 10
                nobject_loss /= 10
                
                print('iter[%d], train loss: %.8f, class_loss: %.8f, coord_loss: %.8f, '
                      'object_loss: %.8f, nobject_loss: %.8f, image_nums: %d, '
                      'speed: %.2f images/s' % (
                    n_iter, train_loss, class_loss, coord_loss, object_loss, nobject_loss, 
                    process_images, speed))
                sys.stdout.flush()
                
                train_loss = 0.0
                class_loss = 0.0
                coord_loss = 0.0
                object_loss = 0.0
                nobject_loss = 0.0
            
            # 每100轮观测一次验证集损失值和准确率
            if n_iter % 100 == 0:
                valid_loss, valid_iou, valid_object, valid_nobject, \
                    valid_nobject, valid_recall = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                for i in range(0, processor.n_valid-batch_size, batch_size):
                    batch_images, batch_class_labels, batch_class_masks, batch_box_labels, \
                        batch_object_masks, _, batch_object_nums = \
                        processor.get_valid_batch(i, batch_size)
                    # 数据增强
                    batch_images = processor.data_augmentation(
                        batch_images, flip=False,
                        crop=False, padding=20, whiten=True)
                    
                    [avg_loss, iou_value, object_value,
                     nobject_value, recall_value] = self.sess.run(
                        fetches=[self.avg_loss,
                                 self.iou_value, self.object_value,
                                 self.nobject_value, self.recall_value],
                        feed_dict={self.images: batch_images, 
                                   self.class_labels: batch_class_labels, 
                                   self.class_masks: batch_class_masks,
                                   self.box_labels: batch_box_labels,
                                   self.object_masks: batch_object_masks,
                                   self.object_nums: batch_object_nums,
                                   self.keep_prob: 1.0})
                    valid_loss += avg_loss
                    valid_iou += iou_value
                    valid_object += object_value
                    valid_nobject += nobject_value
                    valid_recall += recall_value
                    
                valid_loss /= i
                valid_iou /= i
                valid_object /= i
                valid_nobject /= i
                valid_recall /= i
                
                print('iter[%d], valid: iou: %.8f, object: %.8f, nobject: %.8f, '
                      'recall: %.8f' % (
                    n_iter, valid_iou, valid_object, valid_nobject, valid_recall))
                sys.stdout.flush()
            
            # 每500轮保存一次模型
            if n_iter % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (n_iter)))
                
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
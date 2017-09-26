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
                 cell_size=5, box_per_cell=5, object_scale=1, noobject_scale=1,
                 coord_scale=1, batch_size=2, noobject_thresh=0.6,
                 recall_thresh=0.5):
        # 设置参数
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_channel = n_channel
        self.max_objects = max_objects_per_image
        self.cell_size = cell_size
        self.n_boxes = box_per_cell
        self.object_scale = float(object_scale)
        self.noobject_scale = float(noobject_scale)
        self.coord_scale = float(coord_scale)
        self.batch_size = batch_size
        self.noobject_thresh = noobject_thresh
        self.recall_thresh = recall_thresh
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.image_size, self.image_size, self.n_channel], 
            name='images')
        self.box_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.max_objects, 6], 
            name='box_labels')
        self.object_nums = tf.placeholder(
            dtype=tf.int32, shape=[self.batch_size, ],
            name='object_nums')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
        
        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.coord_loss, self.object_loss, self.noobject_loss, \
            self.iou_value, self.object_value, self.nobject_value, self.recall_value = \
            self.calculate_loss(self.logits)
            
        # 目标函数和优化器
        tf.add_to_collection('losses', self.coord_loss)
        tf.add_to_collection('losses', self.object_loss)
        tf.add_to_collection('losses', self.noobject_loss)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        
        # 设置学习率
        lr = tf.cond(tf.less(self.global_step, 100), 
                     lambda: tf.constant(0.001),
                     lambda: tf.cond(tf.less(self.global_step, 50000),
                                     lambda: tf.constant(0.01),
                                     lambda: tf.cond(tf.less(self.global_step, 100000),
                                                     lambda: tf.constant(0.001),
                                                     lambda: tf.constant(0.0001))))
        self.optimizer = tf.train.MomentumOptimizer(
            learning_rate=0.001, momentum=0.9).minimize(
                self.avg_loss, global_step=self.global_step)
        
    def inference(self, images):
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(self.batch_size, self.image_size, self.image_size, self.n_channel), 
            n_size=3, n_filter=16, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv1')
        pool_layer1 = PoolLayer(
            input_shape=(self.batch_size, self.image_size, self.image_size, 16),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')
        
        conv_layer2 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/2), int(self.image_size/2), 16), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv2')
        pool_layer2 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/2), int(self.image_size/2), 32),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        
        conv_layer3 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 32), 
            n_size=3, n_filter=64, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv3')
        pool_layer3 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 64),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        
        conv_layer4 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 64), 
            n_size=3, n_filter=128, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv4')
        pool_layer4 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 128),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool4')
        
        conv_layer5 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 128), 
            n_size=3, n_filter=256, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv5')
        pool_layer5 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 256),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool5')
        
        conv_layer6 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 256), 
            n_size=3, n_filter=512, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv6')
        pool_layer6 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 512),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool6')
        
        conv_layer7 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/64), int(self.image_size/64), 512), 
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/64), int(self.image_size/64), 1024), 
            n_size=3, n_filter=1024, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv8')
        conv_layer9 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/64), int(self.image_size/64), 1024), 
            n_size=1, n_filter=self.n_boxes*5, stride=1, activation='none',
            batch_normal=False, weight_decay=5e-4, name='conv9')
        
        # 数据流
        print('\n%-10s\t%-20s\t%-20s\t%s' % (
            'Name', 'Filter', 'Input', 'Output'))
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
        hidden_pool6 = pool_layer6.get_output(input=hidden_conv6)
        
        hidden_conv7 = conv_layer7.get_output(input=hidden_pool6)
        hidden_conv8 = conv_layer8.get_output(input=hidden_conv7)
        hidden_conv9 = conv_layer9.get_output(input=hidden_conv8)
        
        logits = hidden_conv9
        
        print()
        sys.stdout.flush()
        # 网络输出
        return logits
    
    def calculate_loss(self, logits):
        logits = tf.reshape(
            logits, shape=[self.batch_size, self.cell_size, self.cell_size, 
                           self.n_boxes, 5])
        
        # 获取class_pred和box_pred
        self.box_preds = tf.reshape(
            tf.concat([tf.sigmoid(logits[:,:,:,:,0:2]),
                       logits[:,:,:,:,2:4],
                       tf.sigmoid(logits[:,:,:,:,4:5])], axis=4), 
            shape=[self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 5])
        
        # 循环每一个example
        results = tf.while_loop(
            cond=self._one_example_cond, 
            body=self._one_example_body, 
            loop_vars=[tf.constant(0), self.batch_size,
                       tf.constant(0.0), tf.constant(0.0), tf.constant(0.0),
                       tf.constant(0.0), tf.constant(0.0), tf.constant(0.0), tf.constant(0.0)])
        coord_loss = results[2]
        object_loss = results[3]
        noobject_loss = results[4]
        iou_value = results[5]
        object_value = results[6]
        anyobject_value = results[7]
        recall_value = results[8]
            
        # 目标函数值
        coord_loss = coord_loss * self.coord_scale / self.batch_size
        object_loss = object_loss * self.object_scale / self.batch_size
        noobject_loss = noobject_loss * self.noobject_scale / self.batch_size
        # 观察值
        iou_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        object_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        anyobject_value /= (self.batch_size * self.cell_size * self.cell_size * self.n_boxes)
        recall_value /= tf.reduce_sum(tf.cast(self.object_nums, tf.float32), axis=[0])
        
        return coord_loss, object_loss, noobject_loss, \
            iou_value, object_value, anyobject_value, recall_value
            
    def _one_example_cond(self, example, batch_size, coord_loss, object_loss, noobject_loss,
                          iou_value, object_value, anyobject_value, recall_value):
        
        return example < batch_size
    
    def _one_example_body(self, example, batch_size, coord_loss, object_loss, noobject_loss,
                          iou_value, object_value, anyobject_value, recall_value):
        # 循环每一个object，计算每个box对每个object的iou
        results = tf.while_loop(
            cond=self._one_object_iou_cond, 
            body=self._one_object_iou_body, 
            loop_vars=[example, tf.constant(0), self.object_nums[example],
                       tf.zeros(shape=(self.cell_size, self.cell_size, 
                                       self.n_boxes, self.max_objects))])
        iou_tensor_whole = results[3]
        iou_tensor_max = tf.reduce_max(iou_tensor_whole, 3, keep_dims=True)
        noobject_mask = tf.cast(
            (iou_tensor_max <= self.noobject_thresh), dtype=tf.float32)
        
        # 计算noobject_loss
        noobject_label = tf.zeros(
            shape=(self.cell_size, self.cell_size, self.n_boxes, 1),
            dtype=tf.float32)
        noobject_pred = self.box_preds[example,:,:,:,4:]
        noobject_loss += tf.nn.l2_loss(
            (noobject_label - noobject_pred) * noobject_mask)
        
        # 计算anyobject_value
        anyobject_value += tf.reduce_sum(noobject_pred, axis=[0,1,2,3])
            
        # 循环每一个object，计算coord_loss, object_loss和class_loss
        results = tf.while_loop(
            cond=self._one_object_loss_cond, 
            body=self._one_object_loss_body, 
            loop_vars=[example, tf.constant(0), self.object_nums[example],
                       tf.constant(0.0), tf.constant(0.0), tf.constant(0.0),
                       tf.constant(0.0), tf.constant(0.0)])
        coord_loss += results[3]
        object_loss += results[4]
        iou_value += results[5]
        object_value += results[6]
        recall_value += results[7]
        
        example += 1
        
        return example, batch_size, coord_loss, object_loss, noobject_loss, \
            iou_value, object_value, anyobject_value, recall_value
    
    def _one_object_iou_cond(self, example, num, object_num, iou_tensor_whole):
        
        return num < object_num
    
    def _one_object_iou_body(self, example, num, object_num, iou_tensor_whole):
        # 构造box_label
        # 如果cell中有物体，box_label的每一个box为四个坐标，如果cell中没有物体，则均为0
        cell_x = self.box_labels[example, num, 0]
        cell_y = self.box_labels[example, num, 1]
        box_label = tf.cast(self.box_labels[example,num,2:6], dtype=tf.float32)
        box_label = tf.reshape(box_label, shape=(1, 1, 1, 4))
        box_label = tf.tile(box_label, [self.cell_size, self.cell_size, self.n_boxes, 4])
        
        # 构造box_pred
        # 尺寸为(cell_size, cell_size, n_boxes, 4)
        box_pred = self.get_box_pred(self.box_preds[example,:,:,:,0:4])
        
        iou_tensor = self.calculate_iou(box_pred, box_label)
        padding = tf.cast([[0, 0], [0, 0], [0, 0], [num, self.max_objects-num-1]], dtype=tf.int32)
        iou_tensor = tf.pad(iou_tensor, paddings=padding, mode='CONSTANT')
        
        iou_tensor_whole += iou_tensor
        
        num += 1
        
        return example, num, object_num, iou_tensor_whole
    
    def _one_object_loss_cond(self, example, num, object_num, coord_loss, object_loss, 
                              iou_value, object_value, recall_value):
        
        return num < object_num
    
    def _one_object_loss_body(self, example, num, object_num, coord_loss, object_loss, 
                              iou_value, object_value, recall_value):
        # 构造object_mask
        # 如果cell中有物体，object_mask则为1，如果cell中没有物体，则为0
        cell_x = self.box_labels[example, num, 0]
        cell_y = self.box_labels[example, num, 1]
        object_mask = tf.ones(
            shape=(1, 1), dtype=tf.float32)
        padding = tf.cast([[cell_y, self.cell_size-cell_y-1], 
                           [cell_x, self.cell_size-cell_x-1]], dtype=tf.int32)
        object_mask = tf.reshape(
            tf.pad(object_mask, paddings=padding, mode='CONSTANT'),
            shape=(self.cell_size, self.cell_size, 1, 1))
        
        # 构造box_label
        # 如果cell中有物体，box_label的每一个box则为四个坐标，如果cell中没有物体，则为0
        box_label = tf.cast(self.box_labels[example,num,2:6], dtype=tf.float32)
        box_label = tf.reshape(box_label, shape=(1, 1, 1, 4))
        box_label = tf.tile(box_label, [1, 1, self.n_boxes, 4])
        padding = tf.cast([[cell_y, self.cell_size-cell_y-1], 
                           [cell_x, self.cell_size-cell_x-1],
                           [0, 0], [0, 0]], dtype=tf.int32)
        box_label = tf.pad(box_label, paddings=padding, mode='CONSTANT')
        
        # 构造shift_box_label
        # 如果cell中有物体，shift_box_label的每一个box则为0, 0, w, h，如果cell中没有物体，则为0
        pred_x = tf.zeros(shape=(self.cell_size, self.cell_size, self.n_boxes, 1))
        pred_y = tf.zeros(shape=(self.cell_size, self.cell_size, self.n_boxes, 1))
        shift_box_label = tf.concat([pred_x, pred_y, box_label[:,:,:,2:4]], axis=3)
        
        # 构造new_box_pred
        # x和y的pred
        pred_x = tf.zeros(shape=(self.cell_size, self.cell_size, self.n_boxes, 1))
        pred_y = tf.zeros(shape=(self.cell_size, self.cell_size, self.n_boxes, 1))
        # w的pred
        pred_w = tf.cast([0.73, 0.73, 0.71, 0.76, 0.73], dtype=tf.float32)
        pred_w = tf.reshape(pred_w, shape=(1, 1, self.n_boxes, 1))
        pred_w = tf.tile(pred_w, (self.cell_size, self.cell_size, 1, 1))
        # h的pred
        pred_h = tf.cast([0.12, 0.23, 0.17, 0.65, 0.11], dtype=tf.float32)
        pred_h = tf.reshape(pred_h, shape=(1, 1, self.n_boxes, 1))
        pred_h = tf.tile(pred_h, (self.cell_size, self.cell_size, 1, 1))
        pseudo_box_pred = tf.concat([pred_x, pred_y, pred_w, pred_h], axis=3)
        
        # 计算shift_box_label和new_box_pred的iou，选出最大的iou来计算
        pseudo_iou_tensor = self.calculate_iou(pseudo_box_pred, shift_box_label)
        iou_tensor_max = tf.reduce_max(pseudo_iou_tensor, 2, keep_dims=True)
        iou_tensor_mask = tf.cast(
            (pseudo_iou_tensor >= iou_tensor_max), dtype=tf.float32) * object_mask
        
        # 计算coord_loss
        # coord_pred为box_pred的值，尺寸为(cell_size, cell_size, n_box, 1)
        # 每一个cell中，有object，并且iou最大的那个box的coord_label为真实的label，其余为0，
        # coord_label尺寸为(cell_size, cell_size, n_box, 1)
        coord_label = box_label[:,:,:,0:4] * iou_tensor_mask
        coord_pred = self.get_box_pred(self.box_preds[example,:,:,:,0:4]) * iou_tensor_mask
        coord_loss += tf.nn.l2_loss(coord_label[:,:,:,0:4] - coord_pred[:,:,:,0:4])
        
        # 计算iou_value
        # 每一个cell中，有object，并且iou最大的那个对应的iou
        coord_pred = self.get_box_pred(self.box_preds[example,:,:,:,0:4])
        true_iou_tensor = self.calculate_iou(coord_pred, box_label)
        iou_value += tf.reduce_sum(
            true_iou_tensor * iou_tensor_mask, axis=[0,1,2,3])
            
        # 计算recall_value
        # 每一个cell中，有object，并且iou最大的哪个对应的iou如果大于recall_thresh，则加1
        recall_mask = tf.cast(
            (true_iou_tensor * iou_tensor_mask > self.recall_thresh), dtype=tf.float32)
        recall_value += tf.reduce_sum(recall_mask, axis=[0,1,2,3])
        
        # 计算object_loss
        # object_pred为box_pred的值，尺寸为(cell_size, cell_size, n_box, 1)
        # 每一个cell中，有object，并且iou最大的那个box的object_label为iou，其余为0，
        # object_label尺寸为(cell_size, cell_size, n_box, 1)
        object_label = tf.ones(
            shape=(self.cell_size, self.cell_size, self.n_boxes, 1)) * iou_tensor_mask
        object_pred = self.box_preds[example,:,:,:,4:5] * iou_tensor_mask
        object_loss += tf.nn.l2_loss(object_label - object_pred)
        
        # 计算object_value
        # 每一个cell中，有object，并且iou最大的那个对应的box_pred中的confidence
        object_value += tf.reduce_sum(
            self.box_preds[example,:,:,:,4:5] * iou_tensor_mask, axis=[0,1,2,3])
        
        num += 1
        
        return example, num, object_num, coord_loss, object_loss, \
            iou_value, object_value, recall_value
    
    def get_box_pred(self, box_pred):
        # 计算bx
        offset_x = tf.reshape(tf.range(0, self.cell_size), shape=(1, self.cell_size, 1, 1))
        offset_x = tf.tile(offset_x, (self.cell_size, 1, self.n_boxes, 1))
        offset_x = tf.cast(offset_x, dtype=tf.float32)
        x_pred = (box_pred[:,:,:,0:1] + offset_x) / self.cell_size
        
        # 计算by
        offset_y = tf.reshape(tf.range(0, self.cell_size), shape=(self.cell_size, 1, 1, 1))
        offset_y = tf.tile(offset_y, (1, self.cell_size, self.n_boxes, 1))
        offset_y = tf.cast(offset_y, dtype=tf.float32)
        y_pred = (box_pred[:,:,:,1:2] + offset_y) / self.cell_size
        
        # 计算pw
        prior_w = tf.constant([0.73, 0.73, 0.71, 0.76, 0.73], dtype=tf.float32)
        prior_w = tf.reshape(prior_w, shape=(1, 1, self.n_boxes, 1))
        prior_w = tf.tile(prior_w, (self.cell_size, self.cell_size, 1, 1))
        w_pred = prior_w * tf.exp(box_pred[:,:,:,2:3]) / self.cell_size
        
        # 计算ph
        prior_h = tf.constant([0.12, 0.23, 0.17, 0.65, 0.11], dtype=tf.float32)
        prior_h = tf.reshape(prior_h, shape=(1, 1, self.n_boxes, 1))
        prior_h = tf.tile(prior_h, (self.cell_size, self.cell_size, 1, 1))
        h_pred = prior_h * tf.exp(box_pred[:,:,:,3:4]) / self.cell_size
        
        box_pred = tf.concat([x_pred, y_pred, w_pred, h_pred], axis=3)
        
        return box_pred
              
    def calculate_iou(self, box_pred, box_label):
        box1 = tf.stack([
            box_pred[:,:,:,0] - box_pred[:,:,:,2] / 2.0,
            box_pred[:,:,:,1] - box_pred[:,:,:,3] / 2.0,
            box_pred[:,:,:,0] + box_pred[:,:,:,2] / 2.0,
            box_pred[:,:,:,1] + box_pred[:,:,:,3] / 2.0])
        box1 = tf.transpose(box1, perm=[1, 2, 3, 0])
        box2 = tf.stack([
            box_label[:,:,:,0] - box_label[:,:,:,2] / 2.0,
            box_label[:,:,:,1] - box_label[:,:,:,3] / 2.0,
            box_label[:,:,:,0] + box_label[:,:,:,2] / 2.0,
            box_label[:,:,:,1] + box_label[:,:,:,3] / 2.0])
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
        train_avg_loss, train_coord_loss, \
            train_object_loss, train_noobject_loss = 0.0, 0.0, 0.0, 0.0
        train_iou_value, train_object_value, \
            train_anyobject_value, train_recall_value = 0.0, 0.0, 0.0, 0.0 
        
        for n_iter in range(1, n_iters+1):
            # 训练一个batch，计算从准备数据到训练结束的时间
            start_time = time.time()
            
            # 获取数据并进行数据增强
            batch_image_paths, batch_labels = processor.get_random_batch(
                processor.trainsets, batch_size)
            batch_images, batch_labels = processor.data_augmentation(
                batch_image_paths, batch_labels, mode='train',
                flip=True, whiten=True, resize=True, jitter=0.2)
            batch_box_labels, batch_object_nums = \
                processor.process_batch_labels(batch_labels)
            
            [_, avg_loss, coord_loss, object_loss, noobject_loss,
             iou_value, object_value, anyobject_value, recall_value] = self.sess.run(
                fetches=[self.optimizer, self.avg_loss,
                         self.coord_loss,
                         self.object_loss, self.noobject_loss,
                         self.iou_value, self.object_value,
                         self.nobject_value, self.recall_value], 
                feed_dict={self.images: batch_images,
                           self.box_labels: batch_box_labels,
                           self.object_nums: batch_object_nums,
                           self.keep_prob: 0.5})
                
            end_time = time.time()
            
            train_avg_loss += avg_loss
            train_coord_loss += coord_loss
            train_object_loss += object_loss
            train_noobject_loss += noobject_loss
            train_iou_value += iou_value
            train_object_value += object_value
            train_anyobject_value += anyobject_value
            train_recall_value += recall_value
            
            process_images += batch_size
            speed = 1.0 * batch_size / (end_time - start_time)
                
            # 每1轮训练观测一次train_loss    
            print('{TRAIN} iter[%d], train_loss: %.6f, coord_loss: %.6f, '
                  'object_loss: %.6f, nobject_loss: %.6f, image_nums: %d, '
                  'speed: %.2f images/s' % (
                n_iter, train_avg_loss, train_coord_loss, 
                train_object_loss, train_noobject_loss, process_images, speed))
            sys.stdout.flush()
            
            train_avg_loss, train_coord_loss, \
                train_object_loss, train_noobject_loss = 0.0, 0.0, 0.0, 0.0
            
            # 每1轮观测一次训练集evaluation
            print('{TRAIN} iter[%d], iou: %.6f, object: %.6f, '
                  'anyobject: %.6f, recall: %.6f' % (
                n_iter, train_iou_value, train_object_value, 
                train_anyobject_value, train_recall_value))
            sys.stdout.flush()
            
            train_iou_value, train_object_value, \
                train_anyobject_value, train_recall_value = 0.0, 0.0, 0.0, 0.0 
            
            # 每1000轮观测一次验证集evaluation
            if n_iter % 1000 == 0:
                valid_iou_value, valid_object_value, \
                    valid_nobject_value, valid_recall_value = 0.0, 0.0, 0.0, 0.0 
                
                for i in range(0, processor.n_valid-batch_size, batch_size):
                    
                    # 获取数据并进行数据增强
                    batch_image_paths, batch_labels = processor.get_index_batch(
                        processor.validsets, i, batch_size)
                    batch_images, batch_labels = processor.data_augmentation(
                        batch_image_paths, batch_labels, mode='test',
                        flip=False, whiten=True, resize=True, jitter=0.2)
                    batch_box_labels, batch_object_nums = \
                        processor.process_batch_labels(batch_labels)
                    
                    [iou_value, object_value,
                     nobject_value, recall_value] = self.sess.run(
                        fetches=[self.iou_value, self.object_value,
                                 self.nobject_value, self.recall_value],
                        feed_dict={self.images: batch_images,
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
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
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

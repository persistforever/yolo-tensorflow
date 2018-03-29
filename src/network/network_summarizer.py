# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/03/10
# description: network structure in object detection
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
from src.layer.pool_layer import PoolLayer
from src.layer.dense_layer import DenseLayer


class Network:

    def __init__(self, 
        n_channel, 
        n_classes, 
        image_x_size, 
        image_y_size,
        max_objects, 
        cell_x_size, 
        cell_y_size, 
        pool_mode, 
        box_per_cell, 
        batch_size, 
        object_scale, 
        noobject_scale, 
        coord_scale,
        class_scale,
        noobject_thresh=0.6, 
        recall_thresh=0.5, 
        pred_thresh=0.5, 
        nms_thresh=0.4,
        is_weight_decay=False,
        weight_decay_scale=0.0): 
        
        # 设置参数
        self.n_channel = n_channel
        self.n_classes = n_classes + 1
        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.max_objects = max_objects
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.pool_mode = pool_mode
        self.n_boxes = box_per_cell
        self.batch_size = batch_size
        self.object_scale = float(object_scale)
        self.noobject_scale = float(noobject_scale)
        self.coord_scale = float(coord_scale)
        self.class_scale = float(class_scale)
        self.noobject_thresh = noobject_thresh
        self.recall_thresh = recall_thresh
        self.pred_thresh = pred_thresh
        self.nms_thresh = nms_thresh
        self.is_weight_decay = is_weight_decay
        self.weight_decay_scale = float(weight_decay_scale)

        # 全局变量
        grid_x = numpy.array(range(0, self.cell_x_size), dtype='float32')
        grid_x = numpy.reshape(grid_x, newshape=(1, 1, self.cell_x_size, 1, 1))
        grid_x = numpy.tile(grid_x, (self.batch_size, self.cell_y_size, 1, self.n_boxes, 1))
        self.grid_x = tf.constant(grid_x, dtype=tf.float32)
       
        grid_y = numpy.array(range(0, self.cell_y_size), dtype='float32')
        grid_y = numpy.reshape(grid_y, newshape=(1, self.cell_y_size, 1, 1, 1))
        grid_y = numpy.tile(grid_y, (self.batch_size, 1, self.cell_x_size, self.n_boxes, 1))
        self.grid_y = tf.constant(grid_y, dtype=tf.float32)
        
        prior_w = numpy.array([1.0, 0.8, 0.6, 0.4, 0.2], dtype='float32')
        prior_w = numpy.reshape(prior_w, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_w = numpy.tile(prior_w, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        self.prior_w = tf.constant(prior_w, dtype=tf.float32)
        
        prior_h = numpy.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype='float32')
        prior_h = numpy.reshape(prior_h, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_h = numpy.tile(prior_h, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        self.prior_h = tf.constant(prior_h, dtype=tf.float32)
        
        # 网络结构
        print('\n%-10s\t%-25s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output', 'Field')) 
        self.conv_layer1 = ConvLayer(
            x_size=7, y_size=7, x_stride=2, y_stride=2, n_filter=32, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv1',
            input_shape=(self.image_y_size, self.image_x_size, self.n_channel))
        self.pool_layer1 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool1', prev_layer=self.conv_layer1)
        
        self.conv_layer2 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=96, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv2', prev_layer=self.pool_layer1) 
        self.pool_layer2 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool2', prev_layer=self.conv_layer2)
        
        self.conv_layer3 = ConvLayer(
            x_size=1, y_size=1, x_stride=1, y_stride=1, n_filter=64, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv3', prev_layer=self.pool_layer2)
        self.conv_layer4 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=128, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv4', prev_layer=self.conv_layer3) 
        self.conv_layer5 = ConvLayer(
            x_size=1, y_size=1, x_stride=1, y_stride=1, n_filter=128, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv5', prev_layer=self.conv_layer4) 
        self.conv_layer6 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=256, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv6', prev_layer=self.conv_layer5)
        self.pool_layer3 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool3', prev_layer=self.conv_layer6)
        
        self.conv_layer7 = ConvLayer(
            x_size=1, y_size=1, x_stride=1, y_stride=1, n_filter=128, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv7', prev_layer=self.pool_layer3)
        self.conv_layer8 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=256, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv8', prev_layer=self.conv_layer7) 
        self.conv_layer9 = ConvLayer(
            x_size=1, y_size=1, x_stride=1, y_stride=1, n_filter=256, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv9', prev_layer=self.conv_layer8) 
        self.conv_layer10 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=512, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv10', prev_layer=self.conv_layer9)
        self.pool_layer4 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool4', prev_layer=self.conv_layer10)
        
        self.conv_layer11 = ConvLayer(
            x_size=1, y_size=1, x_stride=1, y_stride=1, n_filter=256, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv11', prev_layer=self.pool_layer4)
        self.conv_layer12 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=256, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv12', prev_layer=self.conv_layer11) 
        self.conv_layer13 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=512, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv13', prev_layer=self.conv_layer12) 
        self.pool_layer5 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool5', prev_layer=self.conv_layer13)
        
        self.conv_layer14 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=512, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv14', prev_layer=self.pool_layer5) 
        self.conv_layer15 = ConvLayer(
            x_size=3, y_size=3, x_stride=1, y_stride=1, n_filter=1024, activation='leaky_relu', 
            batch_normal=True, weight_decay=self.weight_decay_scale, name='conv15', prev_layer=self.conv_layer14) 
        self.pool_layer6 = PoolLayer(
            x_size=2, y_size=2, x_stride=2, y_stride=2, mode=self.pool_mode, resp_normal=False, 
            name='pool6', prev_layer=self.conv_layer15)
        
        self.dense_layer1 = DenseLayer(
            hidden_dim=1024, activation='leaky_relu', batch_normal=False, weight_decay=self.weight_decay_scale, 
            name='dense1', input_shape=[4*4*1024])
        self.dense_layer2 = DenseLayer(
            hidden_dim=self.cell_y_size*self.cell_x_size*self.n_boxes*(5+self.n_classes), 
            activation='none', batch_normal=False, weight_decay=self.weight_decay_scale, 
            name='dense2', prev_layer=self.dense_layer1)
        
        self.layers = [
            self.conv_layer1, self.pool_layer1, 
            self.conv_layer2, self.pool_layer2,
            self.conv_layer3, self.conv_layer4, self.conv_layer5, self.conv_layer6, self.pool_layer3,
            self.conv_layer7, self.conv_layer8, self.conv_layer9, self.conv_layer10, self.pool_layer4,
            self.conv_layer11, self.conv_layer12, self.conv_layer13, self.pool_layer5,
            self.conv_layer14, self.conv_layer15, self.pool_layer6,
            self.dense_layer1, self.dense_layer2]

        self.calculation = sum([layer.calculation for layer in self.layers])
        print('calculation: %.2fM\n' % (self.calculation / 1024.0 / 1024.0))

    def get_loss(self, images, coord_true, object_mask, class_true, 
        unpos_coord_true, unpos_object_mask, object_nums, global_step, name):
        
        self.images = tf.stop_gradient(images)
        self.coord_true = coord_true
        self.object_mask = object_mask
        self.class_true = class_true
        self.unpos_coord_true = unpos_coord_true
        self.unpos_object_mask = unpos_object_mask
        self.object_nums = object_nums
        self.global_step = global_step

        # 待输出的中间变量
        self.logits = self.inference(self.images, is_training=tf.constant(True))
        self.loss, self.noobject_loss, self.object_loss, self.coord_loss, self.class_loss, \
            self.iou_value, self.object_value, self.noobject_value = self.calculate_loss(self.logits)
        self.weight_decay_loss = tf.constant(0.0)
        
        tf.add_to_collection('losses_%s' % (name), self.loss)

        if self.is_weight_decay:
            for layer in self.layers:
                if layer.ltype =='conv' and layer.weight_decay:
                    weight_decay_loss = tf.multiply(tf.nn.l2_loss(layer.weight), layer.weight_decay)
                    self.weight_decay_loss += weight_decay_loss
                    tf.add_to_collection('losses_%s' % (name), weight_decay_loss)
        
        self.avg_loss = tf.add_n(tf.get_collection('losses_%s' % (name)))
        self.noobject_loss /= self.batch_size
        self.object_loss /= self.batch_size
        self.coord_loss /= self.batch_size
        self.class_loss /= self.batch_size
            
        return self.avg_loss, self.noobject_loss, self.object_loss, self.coord_loss, self.class_loss, \
            self.weight_decay_loss, self.iou_value, self.object_value, self.noobject_value

    def get_inference(self, images):
       
        self.logits = self.inference(self.images, is_training=tf.constant(False))
        return self.logits

    def inference(self, images, is_training=True):
        with tf.name_scope('inference'):
            # 数据流
            hidden_state = images
            for layer in self.layers[:-2]:
                hidden_state = layer.get_output(input=hidden_state, is_training=is_training)
            hidden_state = tf.reshape(hidden_state, (self.batch_size, 4*4*1024))
            for layer in self.layers[-2:]:
                hidden_state = layer.get_output(input=hidden_state, is_training=is_training)
            logits = hidden_state
            
            # 网络输出 
            logits = tf.reshape(logits, shape=(
                self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 5+self.n_classes))
            logits1 = tf.sigmoid(tf.reshape(logits[:,:,:,:,0:5], shape=[
                self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 5]))
            logits2 = tf.nn.softmax(tf.reshape(logits[:,:,:,:,5:], shape=[
                self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.n_classes]))
            logits = tf.concat([logits1, logits2], axis=4)
        
        return logits
    
    def calculate_loss(self, logits):
        with tf.name_scope('detection'):
            # 获取class_pred和box_pred
            conf_pred = logits[:,:,:,:,0:1]
            coord_pred = logits[:,:,:,:,1:5]
            class_pred = logits[:,:,:,:,5:5+self.n_classes]

            with tf.name_scope('data'):
                # 获得扩展后的coord_pred和coord_true
                coord_pred_convert = self.get_direct_position(coord_pred)
                coord_pred_iter = tf.tile(
                    tf.reshape(coord_pred_convert, shape=[
                        self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 4]), 
                    [1, 1, 1, 1, self.max_objects, 1]) 
                coord_true_iter = tf.reshape(self.coord_true[:,:,:,:,0:4], shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, 4])
                coord_true_iter = tf.tile(coord_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
                class_true_iter = tf.reshape(self.class_true[:,:,:,:,0:self.n_classes], shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, self.n_classes])
                class_true_iter = tf.tile(class_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
                
                # 获得扩展后的unpos_coord_true
                unpos_coord_true_iter = tf.reshape(self.unpos_coord_true[:,:,0:4], shape=[
                    self.batch_size, 1, 1, 1, self.max_objects, 4])
                unpos_coord_true_iter = tf.tile(unpos_coord_true_iter, [
                    1, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 1])
                
                # 获得shift_coord_true，将x和y改成0，w和h不变
                shift_xy = tf.zeros(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects, 2))
                shift_coord_true = tf.concat(
                    [shift_xy, self.coord_true[:,:,:,:,2:4]], axis=4)
                shift_coord_true_iter = tf.reshape(shift_coord_true, shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, 4))
                shift_coord_true_iter = tf.tile(shift_coord_true_iter, (
                    1, 1, 1, self.n_boxes, 1, 1))
                
                # 获得pseudo_coord_pred，将x和y改成0，w和h变为base
                pseudo_xy = tf.zeros(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 2)) 
                pseudo_coord_pred = tf.concat([pseudo_xy, self.prior_w, self.prior_h], axis=4)
                pseudo_coord_pred_iter = tf.reshape(pseudo_coord_pred, shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 4))
                pseudo_coord_pred_iter = tf.tile(pseudo_coord_pred_iter, (
                    1, 1, 1, 1, self.max_objects, 1))

                # 计算IOU mask, 尺寸为(1,2,2,2,3,1)
                # 根据pseudo_iou_tensor计算得到iou_tensor_pred_mask
                pseudo_iou_tensor = self.calculate_iou(
                    pseudo_coord_pred_iter, shift_coord_true_iter, mode='xywh')
                pseudo_iou_tensor = tf.reshape(pseudo_iou_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, 
                    self.n_boxes, self.max_objects, 1])
                iou_tensor_max = tf.reduce_max(pseudo_iou_tensor, 3, keep_dims=True) 
                iou_tensor_mask = tf.cast((pseudo_iou_tensor >= iou_tensor_max), dtype=tf.float32) 
                iou_tensor_mask *= tf.reshape(self.object_mask, shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, 1))
                iou_tensor_pred_mask = tf.reduce_max(iou_tensor_mask, axis=4)
                # 计算得到iou_tensor
                iou_tensor = self.calculate_iou(coord_pred_iter, coord_true_iter)
                iou_tensor = tf.reshape(iou_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.max_objects, 1])
                
            with tf.name_scope('noobject'):
                # 根据iou_tensor计算得到iou_anyobject_mask
                noobject_mask = tf.ones(
                    (self.batch_size, self.cell_y_size, self.cell_x_size, 
                    self.n_boxes, 1), dtype=tf.float32) - iou_tensor_pred_mask
            
                # 计算anyobject_output
                zeros_label = tf.zeros(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1))
                noobject_output = (zeros_label - conf_pred) * tf.stop_gradient(noobject_mask)
                noobject_loss = self.noobject_scale * tf.nn.l2_loss(noobject_output)

                # 计算noobject_value
                noobject_value = tf.reduce_sum(
                    conf_pred * noobject_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(noobject_mask, axis=[0,1,2,3,4]))
            
            with tf.name_scope('object'):
                # 计算object_output
                ones_label = tf.ones(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1))
                object_output = (ones_label - conf_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                object_loss = self.object_scale * tf.nn.l2_loss(object_output) ** 2

                # 计算object_value
                conf_pred_iter = tf.tile(tf.reshape(conf_pred, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 1]),
                    [1, 1, 1, 1, self.max_objects, 1])
                object_value = tf.reduce_sum(
                    conf_pred * iou_tensor_pred_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(iou_tensor_pred_mask, axis=[0,1,2,3,4]))
            
            with tf.name_scope('coord'):
                # 计算coord_output
                iou_tensor_pred_mask_copy = tf.reshape(iou_tensor_pred_mask, shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 1))
                coord_label_matrix = iou_tensor_max * iou_tensor_pred_mask_copy
                coord_label_max = tf.reduce_max(coord_label_matrix, axis=4, keep_dims=True)
                coord_label_mask = tf.cast(
                    coord_label_matrix >= coord_label_max, dtype=tf.float32) * iou_tensor_pred_mask_copy
                coord_label = tf.reduce_max(coord_label_mask * coord_true_iter, axis=4)
                coord_label = tf.stop_gradient(self.get_inverse_position(coord_label))
                coord_output = (coord_label - coord_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                coord_loss = self.coord_scale * tf.nn.l2_loss(coord_output)

                # 计算iou_value
                iou_value = tf.reduce_sum(
                    iou_tensor * iou_tensor_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))

            with tf.name_scope('class'):
                # 计算class_loss和class_value
                class_label = tf.reduce_max(coord_label_mask * class_true_iter, axis=4)
                class_label = tf.stop_gradient(class_label)
                class_output = (class_label - class_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                class_loss = self.class_scale * tf.nn.l2_loss(class_output)
            
            loss = (noobject_loss + object_loss + coord_loss + class_loss) / self.batch_size

            return loss, noobject_loss, object_loss, coord_loss, class_loss, \
                iou_value, object_value, noobject_value
    
    def get_direct_position(self, coord_pred):
        """
        将相对anchor box的预测框转化为绝对预测框
        输入：相对anchor box的预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        输出：绝对预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        """
        with tf.name_scope('direct_position'):
            x_pred = (coord_pred[:,:,:,:,0:1] + self.grid_x) / self.cell_x_size
            y_pred = (coord_pred[:,:,:,:,1:2] + self.grid_y) / self.cell_y_size
            w_pred = coord_pred[:,:,:,:,2:3]
            h_pred = coord_pred[:,:,:,:,3:4]
            
            new_coord_pred = tf.concat([x_pred, y_pred, w_pred, h_pred], axis=4)
        
        return new_coord_pred
              
    def get_inverse_position(self, coord_true):
        """
        将绝对真实框转化为相对标记框label
        输入：绝对真实框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        输出：相对标记框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        """
        with tf.name_scope('inverse_position'):
            x_pred = (coord_true[:,:,:,:,0:1] * self.cell_x_size - self.grid_x)
            y_pred = (coord_true[:,:,:,:,1:2] * self.cell_y_size - self.grid_y)
            w_pred = coord_true[:,:,:,:,2:3]
            h_pred = coord_true[:,:,:,:,3:4]
            
            coord_label = tf.concat([x_pred, y_pred, w_pred, h_pred], axis=4)
        
        return coord_label

    def calculate_iou(self, box_pred, box_true, mode='xywh'):
        with tf.name_scope('iou'):
            if mode == 'xywh':
                box1 = tf.stack([
                    box_pred[:,:,:,:,:,0] - box_pred[:,:,:,:,:,2] / 2.0,
                    box_pred[:,:,:,:,:,1] - box_pred[:,:,:,:,:,3] / 2.0,
                    box_pred[:,:,:,:,:,0] + box_pred[:,:,:,:,:,2] / 2.0,
                    box_pred[:,:,:,:,:,1] + box_pred[:,:,:,:,:,3] / 2.0])
                box1 = tf.transpose(box1, perm=[1, 2, 3, 4, 5, 0])
                box2 = tf.stack([
                    box_true[:,:,:,:,:,0] - box_true[:,:,:,:,:,2] / 2.0,
                    box_true[:,:,:,:,:,1] - box_true[:,:,:,:,:,3] / 2.0,
                    box_true[:,:,:,:,:,0] + box_true[:,:,:,:,:,2] / 2.0,
                    box_true[:,:,:,:,:,1] + box_true[:,:,:,:,:,3] / 2.0])
                box2 = tf.transpose(box2, perm=[1, 2, 3, 4, 5, 0])
            
            left_top = tf.maximum(box1[:,:,:,:,:,0:2], box2[:,:,:,:,:,0:2])
            right_bottom = tf.minimum(box1[:,:,:,:,:,2:4], box2[:,:,:,:,:,2:4])
            intersection = right_bottom - left_top
            inter_area = intersection[:,:,:,:,:,0] * intersection[:,:,:,:,:,1]
            mask = tf.cast(intersection[:,:,:,:,:,0] > 0, tf.float32) * \
                tf.cast(intersection[:,:,:,:,:,1] > 0, tf.float32)
            inter_area = inter_area * mask
            box1_area = (box1[:,:,:,:,:,2]-box1[:,:,:,:,:,0]) * (box1[:,:,:,:,:,3]-box1[:,:,:,:,:,1])
            box2_area = (box2[:,:,:,:,:,2]-box2[:,:,:,:,:,0]) * (box2[:,:,:,:,:,3]-box2[:,:,:,:,:,1])
            iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        
        return iou

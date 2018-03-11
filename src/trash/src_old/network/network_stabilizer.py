# -*- coding: utf8 -*-
# author: ronniecao
# time: 2017/12/19
# description: network structure in table detection
from __future__ import print_function
import sys
import os
import time
import math
import numpy
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from pdfinsight.ai.yolo_tf.src.layer.conv_layer import ConvLayer
from pdfinsight.ai.yolo_tf.src.layer.pool_layer import PoolLayer


class Network:

    def __init__(self, n_channel, n_classes, image_x_size, image_y_size,
        max_objects_per_image, cell_x_size, cell_y_size, box_per_cell, batch_size, 
        object_scale, noobject_scale, coord_scale, underlap_scale, duplex_scale,
        noobject_thresh=0.6, recall_thresh=0.5, pred_thresh=0.5, nms_thresh=0.4,
        word_inits=None, is_train_word_vector=False, is_underlap=False, 
        is_observe_overlap=False, is_duplex=False):
        
        # 设置参数
        self.n_classes = n_classes
        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.n_channel = n_channel
        self.max_objects = max_objects_per_image
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.n_boxes = box_per_cell
        self.object_scale = math.sqrt(float(object_scale))
        self.noobject_scale = math.sqrt(float(noobject_scale))
        self.coord_scale = math.sqrt(float(coord_scale))
        self.underlap_scale = math.sqrt(float(underlap_scale))
        self.duplex_scale = math.sqrt(float(duplex_scale))
        self.batch_size = batch_size
        self.noobject_thresh = noobject_thresh
        self.recall_thresh = recall_thresh
        self.pred_thresh = pred_thresh
        self.nms_thresh = nms_thresh
        self.word_inits = word_inits
        self.is_train_word_vector = is_train_word_vector
        self.is_underlap = is_underlap
        self.is_observe_overlap = is_observe_overlap
        self.is_duplex = is_duplex
        self.n_coord = 8 if self.is_duplex else 4

        # 全局变量
        grid_x = numpy.array(range(0, self.cell_x_size), dtype='float32')
        grid_x = numpy.reshape(grid_x, newshape=(1, 1, self.cell_x_size, 1, 1))
        grid_x = numpy.tile(grid_x, (self.batch_size, self.cell_y_size, 1, self.n_boxes, 1))
        self.grid_x = tf.constant(grid_x, dtype=tf.float32)
       
        grid_y = numpy.array(range(0, self.cell_y_size), dtype='float32')
        grid_y = numpy.reshape(grid_y, newshape=(1, self.cell_y_size, 1, 1, 1))
        grid_y = numpy.tile(grid_y, (self.batch_size, 1, self.cell_x_size, self.n_boxes, 1))
        self.grid_y = tf.constant(grid_y, dtype=tf.float32)
        
        prior_w = numpy.array([10.1, 9.7, 9.5, 9.4, 9.1], dtype='float32')
        prior_w = numpy.reshape(prior_w, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_w = numpy.tile(prior_w, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        self.prior_w = tf.constant(prior_w, dtype=tf.float32)
        
        prior_h = numpy.array([9.2, 4.2, 2.3, 1.3, 0.5], dtype='float32')
        prior_h = numpy.reshape(prior_h, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_h = numpy.tile(prior_h, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        self.prior_h = tf.constant(prior_h, dtype=tf.float32)
        
        # 词向量矩阵
        if self.word_inits is not None:
            self.word_embeddings = tf.Variable(
                initial_value=self.word_inits, dtype=tf.float32, name='word_embeddings')
            self.has_word_vector = True
        else:
            self.has_word_vector = False

        # 网络结构
        print('\n%-10s\t%-20s\t%-20s\t%s' % ('Name', 'Filter', 'Input', 'Output')) 
        self.conv_layer1 = ConvLayer(
            input_shape=(self.batch_size, self.image_y_size, self.image_x_size, self.n_channel), 
            n_size=3, n_filter=16, stride=1, activation='leaky_relu', 
            batch_normal=True, weight_decay=5e-4, name='conv1')
        self.pool_layer1 = PoolLayer(
            input_shape=(self.batch_size, self.image_y_size, self.image_x_size, 16),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')
        
        self.conv_layer2 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/2), int(self.image_x_size/2), 16), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv2')
        self.pool_layer2 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_y_size/2), int(self.image_x_size/2), 32),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        
        self.conv_layer3 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/4), int(self.image_x_size/4), 32), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv3')
        self.conv_layer4 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/4), int(self.image_x_size/4), 32), 
            n_size=3, n_filter=16, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv4')
        self.conv_layer5 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/4), int(self.image_x_size/4), 16), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv5')
        self.pool_layer3 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_y_size/4), int(self.image_x_size/4), 32),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        
        self.conv_layer6 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/8), int(self.image_x_size/8), 32), 
            n_size=3, n_filter=64, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv6')
        self.conv_layer7 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/8), int(self.image_x_size/8), 64), 
            n_size=3, n_filter=32, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv7')
        self.conv_layer8 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/8), int(self.image_x_size/8), 32), 
            n_size=3, n_filter=64, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv8')
        self.pool_layer4 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_y_size/8), int(self.image_x_size/8), 64),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool4')
        
        self.conv_layer9 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/16), int(self.image_x_size/16), 64), 
            n_size=3, n_filter=128, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv9')
        self.conv_layer10 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/16), int(self.image_x_size/16), 128), 
            n_size=3, n_filter=64, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv10')
        self.conv_layer11 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/16), int(self.image_x_size/16), 64), 
            n_size=3, n_filter=128, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv11')
        self.pool_layer5 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_y_size/16), int(self.image_x_size/16), 128),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool5')
        
        self.conv_layer12 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/32), int(self.image_x_size/32), 128), 
            n_size=3, n_filter=256, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv12')
        self.conv_layer13 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/32), int(self.image_x_size/32), 256), 
            n_size=3, n_filter=128, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv13')
        self.conv_layer14 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/32), int(self.image_x_size/32), 128), 
            n_size=3, n_filter=256, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv14')
        self.pool_layer6 = PoolLayer(
            input_shape=(self.batch_size, int(self.image_y_size/32), int(self.image_x_size/32), 256),
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool6')
        
        self.conv_layer15 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/64), int(self.image_x_size/64), 256), 
            n_size=3, n_filter=256, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv15')
        self.conv_layer16 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/64), int(self.image_x_size/64), 256), 
            n_size=3, n_filter=512, stride=1, activation='leaky_relu',
            batch_normal=True, weight_decay=5e-4, name='conv16')
        self.conv_layer17 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_y_size/64), int(self.image_x_size/64), 512), 
            n_size=1, n_filter=self.n_boxes*(1+self.n_coord), stride=1, activation='none',
            batch_normal=False, weight_decay=5e-4, name='conv17')
       
        self.layers = [
            self.conv_layer1, self.conv_layer2, self.conv_layer3, self.conv_layer4,
            self.conv_layer5, self.conv_layer6, self.conv_layer7, self.conv_layer8,
            self.conv_layer9, self.conv_layer10, self.conv_layer11, self.conv_layer12,
            self.conv_layer13, self.conv_layer14, self.conv_layer15, self.conv_layer16,
            self.conv_layer17, self.pool_layer1, self.pool_layer2, self.pool_layer3,
            self.pool_layer4, self.pool_layer5, self.pool_layer6]
        print()
        sys.stdout.flush()

    def get_loss(self, images, coord_true, object_mask, unpositioned_coord_true, 
        unpositioned_object_mask, global_step, name):

        if self.has_word_vector:
            if images.shape[3] != 1:
                raise('ERROR: images size error!')
            images = tf.reshape(images, (self.batch_size, self.image_y_size, self.image_x_size))
            images = tf.nn.embedding_lookup(self.word_embeddings, images)
            if not self.is_train_word_vector:
                images = tf.stop_gradient(images)

        self.images = images
        self.coord_true = coord_true
        self.object_mask = object_mask
        self.unpositioned_coord_true = unpositioned_coord_true
        self.unpositioned_object_mask = unpositioned_object_mask
        self.global_step = global_step

        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.loss, self.iou_value, self.object_value, self.noobject_value, self.recall_value, \
            self.overlap_value, self.outer_iou_value = self.calculate_loss(self.logits)
        tf.add_to_collection('losses_%s' % (name), self.loss / self.batch_size)
        for layer in self.layers:
            if layer.ltype =='conv' and layer.weight_decay:
                weight_decay = tf.multiply(tf.nn.l2_loss(layer.weight), layer.weight_decay)
                tf.add_to_collection('losses_%s' % (name), weight_decay)
        self.avg_loss = tf.add_n(tf.get_collection('losses_%s' % (name)))
            
        return self.avg_loss, self.iou_value, self.object_value, self.noobject_value, \
            self.recall_value, self.overlap_value, self.outer_iou_value

    def get_inference(self, images):
        if self.has_word_vector:
            if images.shape[3] != 1:
                raise('ERROR: images size error!')
            images = tf.reshape(images, (self.batch_size, self.image_y_size, self.image_x_size))
            images = tf.nn.embedding_lookup(self.word_embeddings, images)
            if not self.is_train_word_vector:
                images = tf.stop_gradient(images)
        self.images = images
        
        self.logits = self.inference(self.images)
        return self.logits

    def inference(self, images):
        with tf.name_scope('inference'):
            # 数据流
            hidden_conv1 = self.conv_layer1.get_output(input=images)
            hidden_pool1 = self.pool_layer1.get_output(input=hidden_conv1)
            
            hidden_conv2 = self.conv_layer2.get_output(input=hidden_pool1)
            hidden_pool2 = self.pool_layer2.get_output(input=hidden_conv2)
            
            hidden_conv3 = self.conv_layer3.get_output(input=hidden_pool2)
            hidden_conv4 = self.conv_layer4.get_output(input=hidden_conv3)
            hidden_conv5 = self.conv_layer5.get_output(input=hidden_conv4)
            hidden_pool3 = self.pool_layer3.get_output(input=hidden_conv5)
            
            hidden_conv6 = self.conv_layer6.get_output(input=hidden_pool3)
            hidden_conv7 = self.conv_layer7.get_output(input=hidden_conv6)
            hidden_conv8 = self.conv_layer8.get_output(input=hidden_conv7)
            hidden_pool4 = self.pool_layer4.get_output(input=hidden_conv8)
            
            hidden_conv9 = self.conv_layer9.get_output(input=hidden_pool4)
            hidden_conv10 = self.conv_layer10.get_output(input=hidden_conv9)
            hidden_conv11 = self.conv_layer11.get_output(input=hidden_conv10)
            hidden_pool5 = self.pool_layer5.get_output(input=hidden_conv11)
            
            hidden_conv12 = self.conv_layer12.get_output(input=hidden_pool5)
            hidden_conv13 = self.conv_layer13.get_output(input=hidden_conv12)
            hidden_conv14 = self.conv_layer14.get_output(input=hidden_conv13)
            hidden_pool6 = self.pool_layer6.get_output(input=hidden_conv14)
            
            hidden_conv15 = self.conv_layer15.get_output(input=hidden_pool6)
            hidden_conv16 = self.conv_layer16.get_output(input=hidden_conv15)
            hidden_conv17 = self.conv_layer17.get_output(input=hidden_conv16)
            logits = hidden_conv17
            
            # 网络输出 
            logits = tf.reshape(logits, shape=[
                self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.n_coord+1])
            if self.is_duplex:
                logits = tf.concat([tf.sigmoid(logits[:,:,:,:,0:1]), tf.sigmoid(logits[:,:,:,:,1:3]), 
                    logits[:,:,:,:,3:5], tf.sigmoid(logits[:,:,:,:,5:9])], axis=4)
            else:
                logits = tf.concat([tf.sigmoid(logits[:,:,:,:,0:1]), tf.sigmoid(logits[:,:,:,:,1:3]), 
                    logits[:,:,:,:,3:5]], axis=4)
        
        return logits
    
    def calculate_loss(self, logits):
        with tf.name_scope('detection'):
            # 获取class_pred和box_pred
            conf_pred = logits[:,:,:,:,0:1]
            coord_pred = logits[:,:,:,:,1:5]
            if self.is_duplex:
                outer_pred = logits[:,:,:,:,5:9]
                outer_true_iter = tf.reshape(self.coord_true[:,:,:,:,4:8], shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, 4])
                outer_true_iter = tf.tile(outer_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
            
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
                # 获得扩展后的unpositioned_coord_true
                unpositioned_coord_true_iter = tf.reshape(self.unpositioned_coord_true[:,:,0:4], shape=[
                    self.batch_size, 1, 1, 1, self.max_objects, 4])
                unpositioned_coord_true_iter = tf.tile(unpositioned_coord_true_iter, [
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
                pseudo_w = self.prior_w / self.cell_x_size
                pseudo_h = self.prior_h / self.cell_y_size
                pseudo_coord_pred = tf.concat([pseudo_xy, pseudo_w, pseudo_h], axis=4)
                pseudo_coord_pred_iter = tf.reshape(pseudo_coord_pred, shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 4))
                pseudo_coord_pred_iter = tf.tile(pseudo_coord_pred_iter, (
                    1, 1, 1, 1, self.max_objects, 1))

                # 计算IOU mask, 尺寸为(1,2,2,2,3,1)
                # 根据pseudo_iou_tensor计算得到iou_tensor_pred_mask
                pseudo_iou_tensor = self.calculate_iou(
                    pseudo_coord_pred_iter, shift_coord_true_iter)
                pseudo_iou_tensor = tf.reshape(pseudo_iou_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.max_objects, 1])
                iou_tensor_max = tf.reduce_max(pseudo_iou_tensor, 3, keep_dims=True) 
                iou_tensor_mask = tf.cast(
                    (pseudo_iou_tensor >= iou_tensor_max), dtype=tf.float32) * tf.reshape(
                        self.object_mask, shape=(
                            self.batch_size, self.cell_y_size, self.cell_x_size, 1, self.max_objects, 1))
                iou_tensor_pred_mask = tf.reduce_max(iou_tensor_mask, axis=4)
                # 计算得到iou_tensor
                iou_tensor = self.calculate_iou(coord_pred_iter, coord_true_iter)
                iou_tensor = tf.reshape(iou_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.max_objects, 1])
                # 根据iou_tensor计算得到iou_recall_mask
                iou_recall_max = tf.reduce_max(iou_tensor, 4) * iou_tensor_pred_mask
                iou_recall_mask = tf.cast(
                    (iou_recall_max >= self.recall_thresh), dtype=tf.float32)

            with tf.name_scope('anyobject'):
                # 根据iou_tensor计算得到iou_anyobject_mask
                iou_anyobject_tensor = self.calculate_iou(
                    coord_pred_iter, unpositioned_coord_true_iter)
                iou_anyobject_tensor = tf.reshape(iou_anyobject_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.max_objects, 1])
                iou_anyobject_tensor *= tf.reshape(self.unpositioned_object_mask, shape=(
                    self.batch_size, 1, 1, 1, self.max_objects, 1))
                iou_anyobject_max = tf.reduce_max(iou_anyobject_tensor, 4)
                iou_anyobject_mask = tf.cast(
                    (iou_anyobject_max <= self.noobject_thresh), dtype=tf.float32)
            
                # 计算anyobject_output
                zeros_label = tf.zeros(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1))
                anyobject_output = self.noobject_scale * (
                    zeros_label - conf_pred) * tf.stop_gradient(iou_anyobject_mask)
                anyobject_output = tf.pad(anyobject_output, paddings=[
                    [0,0], [0,0], [0,0], [0,0], [self.n_coord,0]], mode='CONSTANT')

                # 计算anyobject_value
                noobject_value = tf.reduce_sum(
                    conf_pred, axis=[0,1,2,3,4]) / (
                        self.batch_size * self.cell_y_size * self.cell_x_size * self.n_boxes)
            
            with tf.name_scope('coord'):
                # 计算coord_output
                iou_tensor_pred_mask_copy = tf.reshape(iou_tensor_pred_mask, (
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 1))
                coord_label_matrix = iou_tensor_max * iou_tensor_pred_mask_copy
                coord_label_max = tf.reduce_max(coord_label_matrix, axis=4, keep_dims=True)
                coord_label_mask = tf.cast(
                    coord_label_matrix >= coord_label_max, dtype=tf.float32) * iou_tensor_pred_mask_copy
                coord_label = tf.reduce_max(coord_label_mask * coord_true_iter, axis=4)
                coord_label = self.get_inverse_position(coord_label)
                inner_output = self.coord_scale * (
                    tf.stop_gradient(coord_label) - coord_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                coord_output = tf.pad(inner_output, paddings=[
                    [0,0], [0,0], [0,0], [0,0], [0,1]], mode='CONSTANT')

                # 计算iou_value
                iou_value = tf.reduce_sum(
                    iou_tensor * iou_tensor_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
            outer_iou_value = tf.constant(0.0)
            if self.is_duplex:
                with tf.name_scope('outer'):
                # 外框的限制
                    coord_label = tf.reduce_max(coord_label_mask * coord_true_iter, axis=4)
                    outer_label = tf.reduce_max(coord_label_mask * outer_true_iter, axis=4)
                    outer_label = self.get_inverse_outer_label(coord_label, outer_label)
                    outer_output = self.duplex_scale * (
                        tf.stop_gradient(outer_label) - outer_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                    coord_output = tf.concat([inner_output, outer_output], axis=4)
                    coord_output = tf.pad(coord_output, paddings=[
                        [0,0], [0,0], [0,0], [0,0], [0,1]], mode='CONSTANT')

                # 计算外框IOU
                outer_pred = self.get_direct_outer_label(coord_pred_convert, outer_pred)
                outer_pred_iter = tf.reshape(outer_pred, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 4])
                outer_pred_iter = tf.tile(
                    tf.reshape(outer_pred_iter, shape=[
                        self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 4]), 
                    [1, 1, 1, 1, self.max_objects, 1])
                outer_iou_tensor = self.calculate_iou(outer_pred_iter, outer_true_iter, mode='lrtb')
                outer_iou_tensor = tf.reshape(outer_iou_tensor, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, self.max_objects, 1])

                # 计算外框iou_value
                outer_iou_value = tf.reduce_sum(
                    outer_iou_tensor * iou_tensor_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
            with tf.name_scope('object'):
                # 计算object_output
                ones_label = tf.ones(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1))
                object_output = self.object_scale * (
                    ones_label - conf_pred) * tf.stop_gradient(iou_tensor_pred_mask)
                object_output = tf.pad(object_output, paddings=[
                    [0,0], [0,0], [0,0], [0,0], [self.n_coord,0]], mode='CONSTANT')

                # 计算object_value
                conf_pred_iter = tf.tile(tf.reshape(conf_pred, shape=[
                    self.batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1, 1]),
                    [1, 1, 1, 1, self.max_objects, 1])
                object_value = tf.reduce_sum(
                    conf_pred_iter * iou_tensor_mask, axis=[0,1,2,3,4]) / (
                        tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
                self.object_output = conf_pred * iou_tensor_pred_mask

                # 计算recall_value
                recall_value = tf.reduce_sum(iou_recall_mask, axis=[0,1,2,3,4]) / (
                    tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
            with tf.name_scope('combine'):
                # 合并anyobject_output, coord_output, object_output
                whole_output = anyobject_output
                inv_iou_tensor_pred_mask = tf.ones(shape=(
                    self.batch_size, self.cell_y_size, self.cell_x_size, 
                    self.n_boxes, 1)) - tf.stop_gradient(iou_tensor_pred_mask)
                whole_output = whole_output * inv_iou_tensor_pred_mask + object_output
                whole_output += coord_output
                # 并且计算loss
                loss = tf.nn.l2_loss(whole_output)
          
            overlap_value = tf.constant(0.0)
            if self.is_observe_overlap:
                with tf.name_scope('overlap_punishment'):
                    # 重合度惩罚项，尽量保证所有的预测框相互不重合
                    n_boxes = self.cell_y_size * self.cell_x_size * self.n_boxes

                    # 对logits按照conf_pred进行重新排序
                    new_logits = tf.concat([conf_pred, coord_pred_convert], axis=4)
                    new_logits = tf.reshape(new_logits, shape=(self.batch_size, n_boxes, 5))
                    conf_pred_index = tf.nn.top_k(new_logits[:,:,0], k=n_boxes).indices
                    conf_pred_index = tf.reshape(conf_pred_index, shape=(self.batch_size, n_boxes, 1, 1))
                    conf_pred_index = tf.tile(conf_pred_index, multiples=(1, 1, 5, 1))
                    conf_pred_index = tf.pad(conf_pred_index, paddings=[[0,0], [0,0], [0,0], [1,1]], mode='CONSTANT')
                    batch_index = tf.range(0, self.batch_size, 1)
                    batch_index = tf.reshape(batch_index, shape=(self.batch_size, 1, 1, 1))
                    batch_index = tf.tile(batch_index, (1, n_boxes, 5, 1))
                    batch_index = tf.pad(batch_index, paddings=[[0,0], [0,0], [0,0], [0,2]], mode='CONSTANT')
                    coord_index = tf.range(0, 5, 1)
                    coord_index = tf.reshape(coord_index, shape=(1, 1, 5, 1))
                    coord_index = tf.tile(coord_index, (self.batch_size, n_boxes, 1, 1))
                    coord_index = tf.pad(coord_index, paddings=[[0,0], [0,0], [0,0], [2,0]], mode='CONSTANT')
                    conf_pred_index += batch_index + coord_index
                    logits_resort = tf.gather_nd(new_logits, tf.stop_gradient(conf_pred_index))
                    conf_pred_resort = logits_resort[:,:,0]
                    conf_pred_mask = tf.cast((conf_pred_resort >= self.pred_thresh), dtype=tf.float32)
                    conf_pred_mask = tf.reshape(conf_pred_mask, shape=(self.batch_size, n_boxes, 1))
                    conf_pred_mask = tf.tile(conf_pred_mask, (1, 1, 5))
                    logits_resort = logits_resort * tf.stop_gradient(conf_pred_mask)
                    conf_pred_resort = logits_resort[:,:,0]
                    coord_pred_resort = logits_resort[:,:,1:5]
                    conf_pred_mask = tf.cast((conf_pred_resort >= self.pred_thresh), dtype=tf.float32)

                    # 计算coord_pred_iter，尺寸都是(64, nbox, nbox, 4)
                    coord_pred_itera = tf.reshape(coord_pred_resort, shape=(
                        self.batch_size, n_boxes, 1, 1, 1, 4))
                    coord_pred_itera = tf.tile(coord_pred_itera, (1, 1, n_boxes, 1, 1, 1))
                    coord_pred_iterb = tf.reshape(coord_pred_resort, shape=(
                        self.batch_size, 1, n_boxes, 1, 1, 4))
                    coord_pred_iterb = tf.tile(coord_pred_iterb, (1, n_boxes, 1, 1, 1, 1))
                    # 计算overlap_iou_tensor，尺寸是(64, nbox, nbox)
                    overlap_iou_tensor = self.calculate_iou(coord_pred_itera, coord_pred_iterb)
                    overlap_iou_tensor = tf.reshape(overlap_iou_tensor, shape=(
                        self.batch_size, n_boxes, n_boxes))
                    
                    # 计算出final_conf_pred_mask
                    boxes_mask = tf.zeros(shape=(self.batch_size, n_boxes))
                    conf_pred_result = tf.while_loop(
                        cond=lambda i,n,nb,a,b,c: i < n, body=self.batch_loop_body,
                        loop_vars=[tf.constant(0), self.batch_size, n_boxes, 
                            overlap_iou_tensor, conf_pred_mask, boxes_mask],
                        parallel_iterations=self.batch_size, 
                        shape_invariants=[
                            tf.TensorShape(()), tf.TensorShape(()), tf.TensorShape(()), 
                            tf.TensorShape((self.batch_size, n_boxes, n_boxes)),
                            tf.TensorShape((self.batch_size, n_boxes)),
                            tf.TensorShape((None, n_boxes))])
                    conf_pred_mask = tf.reshape(conf_pred_result[5], shape=(
                        self.batch_size, n_boxes, 1))

                    # 计算overlap_mask，尺寸是(64, nbox, nbox, 1)，表示那些pair需要被计算overlap
                    overlap_mask_itera = tf.reshape(conf_pred_mask, shape=(self.batch_size, n_boxes, 1, 1))
                    overlap_mask_itera = tf.tile(overlap_mask_itera, (1, 1, n_boxes, 1))
                    overlap_mask_iterb = tf.reshape(conf_pred_mask, shape=(self.batch_size, 1, n_boxes, 1))
                    overlap_mask_iterb = tf.tile(overlap_mask_iterb, (1, n_boxes, 1, 1))
                    eye_mask = tf.eye(n_boxes, n_boxes, [self.batch_size])
                    eye_mask = tf.reshape(eye_mask, shape=(self.batch_size, n_boxes, n_boxes, 1))
                    eye_mask = tf.ones(shape=(self.batch_size, n_boxes, n_boxes, 1)) - eye_mask
                    overlap_mask_iter = overlap_mask_itera * overlap_mask_iterb * eye_mask
                    
                    # 计算overlap_IOU_tensor和惩罚值
                    overlap_iou_tensor = tf.reshape(overlap_iou_tensor, shape=(
                        self.batch_size, n_boxes, n_boxes, 1))
                    overlap_iou_tensor = overlap_iou_tensor * tf.stop_gradient(overlap_mask_iter)
                    self.overlap_iou_tensor = overlap_iou_tensor
                    overlap_value = tf.reduce_sum(overlap_iou_tensor) / (
                        tf.reduce_sum(overlap_mask_iter) + 1e-6)
                    
                    if self.is_underlap:
                        overlap_punishment = tf.nn.l2_loss(overlap_iou_tensor) / tf.stop_gradient(
                            tf.reduce_sum(overlap_mask_iter) + 1e-6)
                        satisfied = tf.cond(tf.less(self.global_step, 1000), 
                            lambda: tf.constant(0.0), lambda: tf.constant(1.0))
                        overlap_punishment = satisfied * self.underlap_scale * overlap_punishment
                        loss += overlap_punishment
                        recall_value = overlap_punishment
            
            return loss, iou_value, object_value, noobject_value, recall_value, \
                overlap_value, outer_iou_value
    
    def batch_loop_body(self, index, nbatch, nboxes, overlap_tensor, boxes_mask_matrix, final_boxes_mask_matrix):
        boxes_mask_vector = boxes_mask_matrix[index, :]
        loop_result = tf.while_loop(cond=lambda i,n,a,b: i < n, body=self.nbox_loop_body,
            loop_vars=[tf.constant(0), nboxes, overlap_tensor[index,:,:], boxes_mask_vector])
        new_boxes_mask_vector = tf.reshape(loop_result[3], shape=(1, nboxes))
        new_boxes_mask_matrix = tf.pad(
            new_boxes_mask_vector, paddings=[[index,nbatch-index-1], [0,0]], mode='CONSTANT')
        final_boxes_mask_matrix += new_boxes_mask_matrix
        index += 1
        return index, nbatch, nboxes, overlap_tensor, boxes_mask_matrix, final_boxes_mask_matrix

    def nbox_loop_body(self, index, nboxes, overlap_matrix, boxes_mask_vector):
        overlap_mask_vector = tf.cast(
            overlap_matrix[index,:] >= self.nms_thresh, dtype=tf.float32) * boxes_mask_vector[index]
        self_mask_vector = tf.ones(shape=(nboxes,)) - tf.pad(
            tf.ones(shape=(1,)), paddings=[[index, nboxes-index-1]], mode='CONSTANT')
        boxes_mask_vector *= tf.ones(shape=(nboxes,)) - overlap_mask_vector * self_mask_vector
        index += 1
        return index, nboxes, overlap_matrix, boxes_mask_vector
        
    def get_direct_position(self, coord_pred):
        """
        将相对anchor box的预测框转化为绝对预测框
        输入：相对anchor box的预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        输出：绝对预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        """
        with tf.name_scope('direct_position'):
            x_pred = (coord_pred[:,:,:,:,0:1] + self.grid_x) / self.cell_x_size
            y_pred = (coord_pred[:,:,:,:,1:2] + self.grid_y) / self.cell_y_size
            w_pred = self.prior_w * tf.exp(coord_pred[:,:,:,:,2:3]) / self.cell_x_size
            h_pred = self.prior_h * tf.exp(coord_pred[:,:,:,:,3:4]) / self.cell_y_size
            
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
            w_pred = tf.log(coord_true[:,:,:,:,2:3] * self.cell_x_size / self.prior_w + 1e-12)
            h_pred = tf.log(coord_true[:,:,:,:,3:4] * self.cell_y_size / self.prior_h + 1e-12)
            
            coord_label = tf.concat([x_pred, y_pred, w_pred, h_pred], axis=4)
        
        return coord_label

    def get_direct_outer_label(self, coord_pred, outer_pred):
        """
        将相对内框的offset外框预测值转化为绝对外框预测值
        输入：内框的预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
              相对内框的offset外框预测值，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        输出：绝对外框预测框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        """
        with tf.name_scope('direct_outer_label'):
            new_coord_pred = tf.stack([
                coord_pred[:,:,:,:,0] - coord_pred[:,:,:,:,2] / 2.0, 
                coord_pred[:,:,:,:,0] + coord_pred[:,:,:,:,2] / 2.0,
                coord_pred[:,:,:,:,1] - coord_pred[:,:,:,:,3] / 2.0, 
                coord_pred[:,:,:,:,1] + coord_pred[:,:,:,:,3] / 2.0])
            new_coord_pred = tf.transpose(new_coord_pred, perm=[1, 2, 3, 4, 0])
            new_outer_pred = tf.stack([
                -outer_pred[:,:,:,:,0], outer_pred[:,:,:,:,1],
                -outer_pred[:,:,:,:,2], outer_pred[:,:,:,:,3]])
            new_outer_pred = tf.transpose(new_outer_pred, perm=[1, 2, 3, 4, 0])
            final_outer_pred = new_outer_pred + new_coord_pred

        return final_outer_pred

    def get_inverse_outer_label(self, coord_true, outer_true):
        """
        将绝对外框的预测框转化为相对内框的offset外框预测值
        输入：绝对内部真实框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
              绝对外部真实框，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        输出：相对内框的offset预测值，尺寸(batch_size, cell_y_size, cell_x_size, n_boxes, 4)
        """
        with tf.name_scope('inverse_outer_position'):
            new_coord_true = tf.stack([
                coord_true[:,:,:,:,0] - coord_true[:,:,:,:,2] / 2.0,
                -(coord_true[:,:,:,:,0] + coord_true[:,:,:,:,2] / 2.0),
                coord_true[:,:,:,:,1] - coord_true[:,:,:,:,3] / 2.0,
                -(coord_true[:,:,:,:,1] + coord_true[:,:,:,:,3] / 2.0)])
            new_coord_true = tf.transpose(new_coord_true, perm=[1, 2, 3, 4, 0])
            new_outer_true = tf.stack([
                -outer_true[:,:,:,:,0], outer_true[:,:,:,:,1],
                -outer_true[:,:,:,:,2], outer_true[:,:,:,:,3]])
            new_outer_true = tf.transpose(new_outer_true, perm=[1, 2, 3, 4, 0])
            outer_label = new_outer_true + new_coord_true

        return outer_label
    
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
            elif mode == 'lrtb':
                box1 = tf.stack([
                    box_pred[:,:,:,:,:,0], box_pred[:,:,:,:,:,2],
                    box_pred[:,:,:,:,:,1], box_pred[:,:,:,:,:,3]])
                box1 = tf.transpose(box1, perm=[1, 2, 3, 4, 5, 0])
                box2 = tf.stack([
                    box_true[:,:,:,:,:,0], box_true[:,:,:,:,:,2],
                    box_true[:,:,:,:,:,1], box_true[:,:,:,:,:,3]])
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

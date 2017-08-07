# -*- coding: utf8 -*-
# author: ronniecao
import sys
import os
import math
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.layer.conv_layer import ConvLayer
from src.layer.dense_layer import DenseLayer
from src.layer.pool_layer import PoolLayer

class TinyYolo():
    
    def __init__(self, n_channel=3, n_classes=1, image_size=288, max_objects_per_image=20,
                 cell_size=7, box_per_cell=5, object_scala=1, nobject_scala=1,
                 coord_scala=1, class_scala=1, batch_size=2):
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
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.image_size, self.image_size, self.n_channel], 
            name='images')
        self.class_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, 1], 
            name='class_labels')
        self.class_masks = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size],
            name='class_masks')
        self.box_labels = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects, 5], 
            name='box_labels')
        self.object_masks = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects],
            name='object_masks')
        self.nobject_masks = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects],
            name='nobject_masks')
        self.object_num = tf.placeholder(
            dtype=tf.float32, shape=[self.batch_size, self.max_objects],
            name='object_num')
        self.keep_prob = tf.placeholder(
            dtype=tf.float32, name='keep_prob')
        
        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.class_loss, self.coord_loss, self.object_loss, self.nobject_loss, \
            self.iou_value, self.object_value, self.nobject_value = self.loss(self.logits)
        tf.add_to_collection('losses', (
            self.class_loss + self.coord_loss + self.object_loss + self.nobject_loss))
        # 目标函数和优化器
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(self.avg_loss)
        
    def inference(self, images):
        # 网络结构
        conv_layer1 = ConvLayer(
            input_shape=(self.batch_size, self.image_size, self.image_size, self.n_channel), 
            n_size=3, n_filter=16, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv1')
        pool_layer1 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool1')
        
        conv_layer2 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/2), int(self.image_size/2), 16), 
            n_size=3, n_filter=32, stride=1, activation='relu',
            batch_normal=False, weight_decay=None, name='conv2')
        pool_layer2 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool2')
        
        conv_layer3 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/4), int(self.image_size/4), 32),
            n_size=3, n_filter=64, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv3')
        pool_layer3 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool3')
        
        conv_layer4 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/8), int(self.image_size/8), 64),
            n_size=3, n_filter=128, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv4')
        pool_layer4 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool4')
        
        conv_layer5 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/16), int(self.image_size/16), 128),
            n_size=3, n_filter=256, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv5')
        pool_layer5 = PoolLayer(
            n_size=2, stride=2, mode='max', resp_normal=False, name='pool5')
        
        conv_layer6 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 256),
            n_size=3, n_filter=512, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv6')
        conv_layer7 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 512),
            n_size=3, n_filter=1024, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv7')
        conv_layer8 = ConvLayer(
            input_shape=(self.batch_size, int(self.image_size/32), int(self.image_size/32), 1024),
            n_size=3, n_filter=1024, stride=1, activation='relu', 
            batch_normal=False, weight_decay=None, name='conv8')
        
        dense_layer1 = DenseLayer(
            input_shape=(self.batch_size, int(self.image_size/32) * int(self.image_size/32) * 1024), 
            hidden_dim=self.cell_size * self.cell_size * (self.n_classes + self.n_boxes * 5), 
            activation='sigmoid', dropout=False, keep_prob=None,
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
        hidden_conv7 = conv_layer7.get_output(input=hidden_conv6)
        hidden_conv8 = conv_layer8.get_output(input=hidden_conv7)
        input_dense1 = tf.reshape(hidden_conv8, shape=[
            -1, int(self.image_size/32) * int(self.image_size/32) * 1024])
        output = dense_layer1.get_output(input=input_dense1)
        
        # 网络输出
        return output
    
    def loss(self, logits):
        logits = tf.reshape(
            logits, shape=[self.batch_size, self.cell_size, self.cell_size, 
                           self.n_classes + self.n_boxes * 5])
        
        class_preds = logits[:,:,:,0:self.n_classes]
        box_preds = tf.reshape(
            logits[:,:,:,self.n_classes:], 
            shape=[self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 5])
        
        class_loss = 0.0
        coord_loss = 0.0
        object_loss = 0.0
        nobject_loss = 0.0
        iou_value = 0.0
        object_value = 0.0
        nobject_value = 0.0
        
        for i in range(self.batch_size):
            for j in range(self.max_objects):
                # 计算class_label
                class_pred = class_preds[i,:,:,:]
                class_label = self.class_labels[i,:,:,:]
                class_mask = tf.reshape(
                    self.class_masks[i,:,:], 
                    shape=[self.cell_size, self.cell_size, 1])
                # 计算class_loss
                class_loss += self.object_num[i,j] * self.class_scala * tf.nn.l2_loss(
                    (class_pred - class_label) * class_mask)
                
                # iou_matrix对每一个cell中每一个pred_box对box label求iou，尺寸为(n_cell, n_cell, n_box, 1)
                # 而iou_matrix_true对每一个cell中求出iou最大的pred_box对应的iou，尺寸为(n_cell, n_cell, 1, 1)
                iou_matrix = self.iou(box_preds[i,:,:,:,0:4], self.box_labels[i,:,:,j:j+1,0:4])
                iou_matrix_max = tf.reduce_max(iou_matrix, 2, keep_dims=True)
                iou_matrix_mask = tf.cast((iou_matrix >= iou_matrix_max), dtype=tf.float32)
                
                # 计算confidence_label
                confidence_pred = box_preds[i,:,:,:,4:]
                confidence_label = self.box_labels[i,:,:,j:j+1,4:]
                object_mask = tf.reshape(
                    self.object_masks[i,:,:,j:j+1],
                    shape=[self.cell_size, self.cell_size, 1, 1])
                nobject_mask = tf.reshape(
                    self.nobject_masks[i,:,:,j:j+1],
                    shape=[self.cell_size, self.cell_size, 1, 1])
                # 计算object_loss和nobject_loss
                iou_object_mask = iou_matrix_mask
                iou_nobject_mask = tf.ones_like(
                    iou_matrix_mask, dtype=tf.float32) - iou_matrix_mask
                object_loss += self.object_num[i,j] * self.object_scala * tf.nn.l2_loss(
                    (confidence_pred - confidence_label) * iou_object_mask * object_mask)
                nobject_loss += self.object_num[i,j] * self.nobject_scala * tf.nn.l2_loss(
                    confidence_pred * iou_nobject_mask * nobject_mask)
                
                # 计算coord_label
                x_pred = box_preds[i,:,:,:,0:1]
                x_label = self.box_labels[i,:,:,j:j+1,0:1]
                y_pred = box_preds[i,:,:,:,1:2]
                y_label = self.box_labels[i,:,:,j:j+1,1:2]
                w_pred = box_preds[i,:,:,:,2:3]
                w_label = self.box_labels[i,:,:,j:j+1,2:3]
                h_pred = box_preds[i,:,:,:,3:4]
                h_label = self.box_labels[i,:,:,j:j+1,3:4]
                # 计算coord_loss
                x_loss = self.coord_scala * tf.nn.l2_loss(
                    (x_pred - x_label) * iou_object_mask * object_mask)
                y_loss = self.coord_scala * tf.nn.l2_loss(
                    (y_pred - y_label) * iou_object_mask * object_mask)
                w_loss = self.coord_scala * tf.nn.l2_loss(
                    (tf.sqrt(w_pred) - tf.sqrt(w_label)) * iou_object_mask * object_mask)
                h_loss = self.coord_scala * tf.nn.l2_loss(
                    (tf.sqrt(h_pred) - tf.sqrt(h_label)) * iou_object_mask * object_mask)
                coord_loss += self.object_num[i,j] * (x_loss + y_loss + w_loss + h_loss)
                
                # 计算观察值
                iou_value += self.object_num[i,j] * (tf.reduce_sum(
                    iou_matrix * iou_object_mask * object_mask, axis=[0,1,2]) + 1e-6) \
                    / (tf.reduce_sum(object_mask, axis=[0,1,2]) + 1e-6)
                object_value += self.object_num[i,j] * (tf.reduce_sum(
                    confidence_pred * iou_object_mask * object_mask, axis=[0,1,2]) + 1e-6) \
                    / (tf.reduce_sum(object_mask, axis=[0,1,2]) + 1e-6)
                nobject_value += self.object_num[i,j] * (tf.reduce_sum(
                    confidence_pred * iou_nobject_mask * nobject_mask, axis=[0,1,2]) + 1e-6) \
                    / (tf.reduce_sum(nobject_mask, axis=[0,1,2]) + 1e-6)
        
        # 目标函数值
        class_loss /= self.batch_size
        coord_loss /= self.batch_size
        object_loss /= self.batch_size
        nobject_loss /= self.batch_size
        # 观察值
        iou_value /= self.batch_size
        object_value /= self.batch_size
        nobject_value /= self.batch_size
        
        return class_loss, coord_loss, object_loss, nobject_loss, \
            iou_value, object_value, nobject_value
              
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
        area = intersection[:,:,:,0] * intersection[:,:,:,1]
        mask = tf.cast(intersection[:,:,:,0] > 0, tf.float32) * \
            tf.cast(intersection[:,:,:,1] > 0, tf.float32)
        area = area * mask
        box1_area = (box1[:,:,:,2] - box1[:,:,:,0]) * (box1[:,:,:,3] - box1[:,:,:,1])
        box2_area = (box2[:,:,:,2] - box2[:,:,:,0]) * (box2[:,:,:,3] - box2[:,:,:,1])
        iou = area / (box1_area + box2_area + 1e-6)
        return tf.reshape(iou, shape=[self.cell_size, self.cell_size, self.n_boxes, 1])
    
    def _process_labels_cpu(self, labels):
        # true label and mask in 类别标记
        class_labels = numpy.zeros(
            shape=(labels.shape[0], self.cell_size, self.cell_size, self.n_classes), 
            dtype='int32')
        class_masks = numpy.zeros(
            shape=(labels.shape[0], self.cell_size, self.cell_size),
            dtype='float32')
        
        # true_label and mask in 包围框标记
        box_labels = numpy.zeros(
            shape=(labels.shape[0], self.cell_size, self.cell_size, self.max_objects, 5),
            dtype='float32')
        
        object_masks = numpy.zeros(
            shape=(labels.shape[0], self.cell_size, self.cell_size, self.max_objects), 
            dtype='float32')
        nobject_masks = numpy.ones(
            shape=(labels.shape[0], self.cell_size, self.cell_size, self.max_objects), 
            dtype='float32')
        
        object_num = numpy.zeros(
            shape=(labels.shape[0], self.max_objects), 
            dtype='int32')
        
        for i in range(labels.shape[0]):
            for j in range(self.max_objects):
                
                [center_x, center_y, w, h, class_index] = labels[i,j,:]
                
                if class_index != 0:    
                    # 计算包围框标记
                    center_cell_x = math.floor(self.cell_size * center_x)
                    center_cell_y = math.floor(self.cell_size * center_y)
                    box_labels[i, center_cell_x, center_cell_y, j, :] = numpy.array(
                        [center_x, center_y, w, h] + [1.0])
                    object_masks[i, center_cell_x, center_cell_y, j] = 1.0
                    nobject_masks[i, center_cell_x, center_cell_y, j] = 0.0
                    
                    # 计算类别标记
                    left_cell_x = math.floor(self.cell_size * (center_x - w / 2.0))
                    right_cell_x = math.floor(self.cell_size * (center_x + w / 2.0))
                    top_cell_y = math.floor(self.cell_size * (center_y - h / 2.0))
                    bottom_cell_y = math.floor(self.cell_size * (center_y + h / 2.0))
                    for x in range(left_cell_x, right_cell_x+1):
                        for y in range(top_cell_y, bottom_cell_y+1):
                            _class_label = numpy.zeros(
                                shape=[self.n_classes,], dtype='int32')
                            _class_label[int(class_index)-1] = 1
                            class_labels[i, x, y, :] = _class_label
                            class_masks[i, x, y] = 1.0
                    
                    # object_num增加
                    object_num[i, j] = 1.0
                            
        return class_labels, class_masks, box_labels, object_masks, nobject_masks, object_num
        
    def train(self, processor, backup_path, n_epoch=5, batch_size=128):
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
            # 数据处理
            train_images = processor.train_images
            train_class_labels, train_class_masks, train_box_labels, \
                train_object_masks, train_nobject_masks, train_object_num = \
                    self._process_labels_cpu(processor.train_labels)
            valid_images = processor.train_images
            valid_class_labels, valid_class_masks, valid_box_labels, \
                valid_object_masks, valid_nobject_masks, valid_object_num = \
                    self._process_labels_cpu(processor.valid_labels)
                
            # 开始本轮的训练
            for i in range(0, processor.n_train-batch_size, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_class_labels = train_class_labels[i: i+batch_size]
                batch_class_masks = train_class_masks[i: i+batch_size]
                batch_box_labels = train_box_labels[i: i+batch_size]
                batch_object_masks = train_object_masks[i: i+batch_size]
                batch_nobject_masks = train_nobject_masks[i: i+batch_size]
                batch_object_num = train_object_num[i: i+batch_size]
                
                [_, avg_loss] = self.sess.run(
                    fetches=[self.optimizer, self.avg_loss], 
                    feed_dict={self.images: batch_images, 
                               self.class_labels: batch_class_labels, 
                               self.class_masks: batch_class_masks,
                               self.box_labels: batch_box_labels,
                               self.object_masks: batch_object_masks,
                               self.nobject_masks: batch_nobject_masks,
                               self.object_num: batch_object_num,
                               self.keep_prob: 0.5})
                
            # 在训练之后，获得本轮的训练集损失值和准确率
            train_loss, train_iou, train_object, train_nobject = 0.0, 0.0, 0.0, 0.0
            for i in range(0, processor.n_train-batch_size, batch_size):
                batch_images = train_images[i: i+batch_size]
                batch_class_labels = train_class_labels[i: i+batch_size]
                batch_class_masks = train_class_masks[i: i+batch_size]
                batch_box_labels = train_box_labels[i: i+batch_size]
                batch_object_masks = train_object_masks[i: i+batch_size]
                batch_nobject_masks = train_nobject_masks[i: i+batch_size]
                batch_object_num = train_object_num[i: i+batch_size]
                
                [avg_loss, iou_value, object_value, nobject_value] = self.sess.run(
                    fetches=[self.avg_loss, 
                             self.iou_value, 
                             self.object_value, 
                             self.nobject_value], 
                    feed_dict={self.images: batch_images, 
                               self.class_labels: batch_class_labels, 
                               self.class_masks: batch_class_masks,
                               self.box_labels: batch_box_labels,
                               self.object_masks: batch_object_masks,
                               self.nobject_masks: batch_nobject_masks,
                               self.object_num: batch_object_num,
                               self.keep_prob: 1.0})
                train_loss += avg_loss
                train_iou += iou_value
                train_object += object_value
                train_nobject += nobject_value
                
            train_loss /= i
            train_iou /= i
            train_object /= i
            train_nobject /= i
            
            # 在训练之后，获得本轮的验证集损失值和准确率
            valid_loss, valid_iou, valid_object, valid_nobject = 0.0, 0.0, 0.0, 0.0
            for i in range(0, processor.n_valid-batch_size, batch_size):
                batch_images = valid_images[i: i+batch_size]
                batch_class_labels = valid_class_labels[i: i+batch_size]
                batch_class_masks = valid_class_masks[i: i+batch_size]
                batch_box_labels = valid_box_labels[i: i+batch_size]
                batch_object_masks = valid_object_masks[i: i+batch_size]
                batch_nobject_masks = valid_nobject_masks[i: i+batch_size]
                batch_object_num = valid_object_num[i: i+batch_size]
                
                [avg_loss, iou_value, object_value, nobject_value] = self.sess.run(
                    fetches=[self.avg_loss, 
                             self.iou_value, 
                             self.object_value, 
                             self.nobject_value], 
                    feed_dict={self.images: batch_images, 
                               self.class_labels: batch_class_labels, 
                               self.class_masks: batch_class_masks,
                               self.box_labels: batch_box_labels,
                               self.object_masks: batch_object_masks,
                               self.nobject_masks: batch_nobject_masks,
                               self.object_num: batch_object_num,
                               self.keep_prob: 1.0})
                valid_loss += avg_loss * self.batch_size
                valid_iou += iou_value * self.batch_size
                valid_object += object_value * self.batch_size
                valid_nobject += nobject_value * self.batch_size
                
            valid_loss /= i
            valid_iou /= i
            valid_object /= i
            valid_nobject /= i
            print('epoch[%d], train loss: %.6f, valid: iou: %.6f, object: %.6f, nobject: %.6f' % (
                epoch, train_loss, valid_iou, valid_object, valid_nobject))
            sys.stdout.flush()
            
            # 保存模型
            saver_path = self.saver.save(
                self.sess, os.path.join(backup_path, 'model.ckpt'))
            if epoch <= 100 and epoch % 10 == 0 or epoch <= 1000 and epoch % 100 == 0 or \
                epoch <= 10000 and epoch % 1000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model_%d.ckpt' % (epoch)))
        self.sess.close()
                
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
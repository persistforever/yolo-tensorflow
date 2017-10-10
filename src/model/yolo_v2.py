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
    
    def __init__(self, n_channel, n_classes, image_size, max_objects_per_image,
                 cell_size, box_per_cell, object_scale, noobject_scale,
                 coord_scale, class_scale, batch_size, noobject_thresh=0.6,
                 recall_thresh=0.5, pred_thresh=0.5, nms_thresh=0.4):
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
        self.class_scale = float(class_scale)
        self.batch_size = batch_size
        self.noobject_thresh = noobject_thresh
        self.recall_thresh = recall_thresh
        self.pred_thresh = pred_thresh
        self.nms_thresh = nms_thresh
        
        # 输入变量
        self.images = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.image_size, self.image_size, self.n_channel], 
            name='images')
        self.coord_true = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects, 4], 
            name='coord_true')
        self.class_true = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects, self.n_classes], 
            name='class_true')
        self.object_mask = tf.placeholder(
            dtype=tf.float32, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.max_objects], 
            name='object_mask')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
        
        # 待输出的中间变量
        self.logits = self.inference(self.images)
        self.coord_loss, self.object_loss, self.noobject_loss, self.class_loss, \
            self.iou_value, self.object_value, self.nobject_value, \
            self.recall_value, self.class_value = \
            self.calculate_loss(self.logits)
            
        # 目标函数和优化器
        tf.add_to_collection('losses', self.coord_loss)
        tf.add_to_collection('losses', self.object_loss)
        tf.add_to_collection('losses', self.noobject_loss)
        tf.add_to_collection('losses', self.class_loss)
        self.avg_loss = tf.add_n(tf.get_collection('losses'))
        
        # 设置学习率
        lr = tf.cond(tf.less(self.global_step, 100),
                     lambda: tf.constant(0.001),
                     lambda: tf.cond(tf.less(self.global_step, 80000),
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
            n_size=1, n_filter=self.n_boxes*(5+self.n_classes), stride=1, activation='none',
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
        
        logits = tf.reshape(logits, shape=[
            self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 5+self.n_classes])
        logits = tf.concat(
            [tf.sigmoid(logits[:,:,:,:,0:5]), tf.nn.softmax(logits[:,:,:,:,5:])], axis=4)
        
        return logits
    
    def calculate_loss(self, logits):
        # 获取class_pred和box_pred
        coord_pred = logits[:,:,:,:,0:4]
        conf_pred = logits[:,:,:,:,4:5]
        class_pred = logits[:,:,:,:,5:]
        
        # 计算每个box_pred和box_trued的IOU
        coord_pred_iter = tf.tile(
            tf.reshape(coord_pred, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 1, 4]), 
            [1, 1, 1, 1, self.max_objects, 1])
        
        coord_true_iter = tf.reshape(self.coord_true, shape=[
                self.batch_size, self.cell_size, self.cell_size, 1, self.max_objects, 4])
        coord_true_iter = tf.tile(coord_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
        
        class_true_iter = tf.reshape(self.class_true, shape=[
            self.batch_size, self.cell_size, self.cell_size, 1, self.max_objects, self.n_classes])
        class_true_iter = tf.tile(class_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
        
        # 计算IOU mask, 尺寸为(1,2,2,2,3,1)
        iou_tensor = self.calculate_iou_tf(coord_pred_iter, coord_true_iter)
        iou_tensor_max = tf.reduce_max(iou_tensor, 3, keep_dims=True) 
        iou_tensor_mask = tf.cast(
            (iou_tensor >= iou_tensor_max), dtype=tf.float32) * tf.reshape(
                self.object_mask, shape=(
                    self.batch_size, self.cell_size, self.cell_size, 1, self.max_objects, 1))
        iou_tensor_pred_mask = tf.reduce_sum(iou_tensor_mask, axis=4)
        
        # 获得coord_label，预测的coord_pred应该接近coord_label，尺寸为(1,2,2,2,4)
        coord_label = tf.reduce_max(iou_tensor_mask * coord_true_iter, axis=4)
        coord_loss = self.coord_scale * tf.nn.l2_loss(
            (coord_pred - coord_label) * iou_tensor_pred_mask)  / (
            tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
        # 获得iou_value，iou最大位置的IOU值之和
        iou_value = tf.reduce_sum(
            tf.reduce_max(iou_tensor, axis=4) * iou_tensor_pred_mask, axis=[0,1,2,3]) / (
                tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
        
        # 获得conf_label，预测的conf_pred应该接近conf_label，尺寸为(1,2,2,2,1)
        # 计算object_loss只计算iou_tensor_mask中1的位置
        conf_label = tf.reduce_max(iou_tensor_mask * tf.ones(shape=(
            self.batch_size, self.cell_size, self.cell_size, 
            self.n_boxes, self.max_objects, 1)), axis=4)
        object_loss = self.object_scale * tf.nn.l2_loss(
            (conf_pred - conf_label) * iou_tensor_pred_mask) / (
            tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
        # 获得object_value，iou最大位置的confidence之和
        object_value = tf.reduce_sum(
            conf_pred * iou_tensor_pred_mask, axis=[0,1,2,3]) / (
                tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
        # 计算noobject_loss只计算iou_tensor_mask中0的位置
        inv_iou_tensor_pred_mask = tf.ones(shape=(
            self.batch_size, self.cell_size, self.cell_size, 
            self.n_boxes, 1)) - iou_tensor_pred_mask
        noobject_loss = self.noobject_scale * tf.nn.l2_loss(
            (conf_pred - conf_label) * inv_iou_tensor_pred_mask) / (
            tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
        # 获得noobject_value，iou最大位置取反后的confidence之和
        noobject_value = tf.reduce_sum(
            conf_pred * inv_iou_tensor_pred_mask, axis=[0,1,2,3]) / (
                tf.reduce_sum(inv_iou_tensor_pred_mask, axis=[0,1,2,3]))
        
        # 获得class_label，预测的class_pred应该接近class_label，尺寸为(1,2,2,2,10)
        class_label = tf.reduce_max(iou_tensor_mask * class_true_iter, axis=4)
        class_loss = self.class_scale * tf.nn.l2_loss(
            (class_pred - class_label) * iou_tensor_pred_mask) / (
            tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
        # 获得class_value，iou最大位置的class最大值
        class_value = tf.reduce_sum(
            class_pred * class_label, axis=[0,1,2,3,4]) / (
                tf.reduce_sum(self.object_mask, axis=[0,1,2,3]))
            
        recall_value = tf.ones((1,))
        tf.reduce_sum(recall_value, axis=0)
            
        return coord_loss, object_loss, noobject_loss, class_loss, \
            iou_value, object_value, noobject_value, recall_value, class_value
              
    def calculate_iou_tf(self, box_pred, box_true):
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
        
        return tf.reshape(iou, shape=[
            self.batch_size, self.cell_size, self.cell_size, self.n_boxes, self.max_objects, 1])
    
    def calculate_iou_py(self, box_pred, box_label):
        box1 = [box_pred[0] - box_pred[2] / 2.0,
                box_pred[1] - box_pred[3] / 2.0,
                box_pred[0] + box_pred[2] / 2.0,
                box_pred[1] + box_pred[3] / 2.0]
        box2 = [box_label[0] - box_label[2] / 2.0,
                box_label[1] - box_label[3] / 2.0,
                box_label[0] + box_label[2] / 2.0,
                box_label[1] + box_label[3] / 2.0]
        left = max(box1[0], box2[0])
        top = max(box1[1], box2[1])
        right = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        inter_area = (right - left) * (bottom - top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6) if inter_area >= 0 else 0.0
        
        return iou
        
    def train(self, dataset, backup_path, n_iters=500000, batch_size=128):
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
            train_object_loss, train_noobject_loss, train_class_loss = \
            0.0, 0.0, 0.0, 0.0, 0.0
        train_iou_value, train_object_value, \
            train_anyobject_value, train_recall_value, train_class_value = \
            0.0, 0.0, 0.0, 0.0, 0.0
        
        start_time = time.time()
        for n_iter in range(1, n_iters+1):
            # 获取数据
            [batch_images, batch_coord_true, batch_class_true, batch_object_mask] = dataset.get()

            [_, avg_loss, coord_loss, object_loss, noobject_loss, class_loss,
             iou_value, object_value, anyobject_value, recall_value, class_value] = \
                self.sess.run(
                    fetches=[self.optimizer, self.avg_loss, self.coord_loss, 
                             self.object_loss, self.noobject_loss, self.class_loss, 
                             self.iou_value, self.object_value, self.nobject_value, 
                             self.recall_value, self.class_value], 
                    feed_dict={self.images: batch_images, 
                               self.coord_true: batch_coord_true,
                               self.class_true: batch_class_true, 
                               self.object_mask: batch_object_mask,
                               self.keep_prob: 0.5})
            
            train_avg_loss += avg_loss
            train_coord_loss += coord_loss
            train_object_loss += object_loss
            train_noobject_loss += noobject_loss
            train_class_loss += class_loss
            train_iou_value += iou_value
            train_object_value += object_value
            train_anyobject_value += anyobject_value
            train_recall_value += recall_value
            train_class_value += class_value
                
            process_images += batch_size
            
            end_time = time.time()
            spend = end_time - start_time
                
            # 每1轮训练观测一次train_loss    
            print('{TRAIN} [%d], train_loss: %.6f, coord_loss: %.6f, '
                  'object_loss: %.6f, nobject_loss: %.6f, class_loss: %.6f, '
                  'image_nums: %d, time: %.2f' % (
                n_iter, train_avg_loss, train_coord_loss, train_object_loss, 
                train_noobject_loss, train_class_loss, process_images, spend))
            sys.stdout.flush()
            
            train_avg_loss, train_coord_loss, train_object_loss, \
                train_noobject_loss, train_class_loss = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 每1轮观测一次训练集evaluation
            print('{TRAIN} [%d], IOU: %.6f, Object: %.6f, '
                  'Noobject: %.6f, Recall: %.6f, Class: %.6f\n' % (
                n_iter, train_iou_value, train_object_value, 
                train_anyobject_value, train_recall_value, train_class_value))
            sys.stdout.flush()
            
            train_iou_value, train_object_value, train_anyobject_value, \
                train_recall_value, train_class_value = 0.0, 0.0, 0.0, 0.0, 0.0
            
            # 每10000轮保存一次模型
            if n_iter % 10000 == 0:
                saver_path = self.saver.save(
                    self.sess, os.path.join(backup_path, 'model.ckpt'))
            
        self.sess.close()
                
    def test(self, processor, backup_dir, output_dir, batch_size=128):
        # 构建会话
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        model_path = os.path.join(backup_dir, 'model.ckpt')
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.sess, model_path)
        print('read model from %s' % (model_path))
        
        # 获取数据并进行数据增强
        batch_image_paths, batch_labels = processor.get_index_batch(
            processor.testsets, 0, batch_size)
        batch_images, _ = processor.data_augmentation(
            batch_image_paths, batch_labels, mode='test',
            flip=False, whiten=True, resize=True)
        
        [logits] = self.sess.run(
            fetches=[self.logits], 
            feed_dict={self.images: batch_images,
                       self.keep_prob: 1.0})
            
        box_preds = numpy.reshape(
            logits, (self.batch_size, self.cell_size, self.cell_size, 
                     self.n_boxes, 5+self.n_classes))
    
        for j in range(batch_images.shape[0]):
            image_path = batch_image_paths[j]
            output_path = os.path.join(output_dir, os.path.split(image_path)[1])
            image = cv2.imread(image_path)
            
            # 获得预测的preds
            preds = []
            for x in range(self.cell_size):
                for y in range(self.cell_size):
                    for n in range(self.n_boxes):
                        box = box_preds[j,x,y,n,0:4]
                        prob = box_preds[j,x,y,n,4] * max(box_preds[j,x,y,n,5:])
                        index = numpy.argmax(box_preds[j,x,y,n,5:])
                        if prob >= self.pred_thresh:
                            preds.append([box, index, prob])
            
            # 排序并去除box
            preds = sorted(preds, key=lambda x: x[2], reverse=True)
            for x in range(len(preds)):
                for y in range(x+1, len(preds)):
                    iou = self.calculate_iou_py(preds[x][0], preds[y][0])
                    if preds[x][1] == preds[y][1] and iou > self.nms_thresh:
                        preds[y][2] = 0.0
            
            # 画预测的框
            for k in range(len(preds)):
                if preds[k][2] > self.pred_thresh:
                    box = preds[k][0]
                    left = int(min(max(0.0, (box[0] - box[2] / 2.0)), 0.9999) * image.shape[1])
                    right = int(min(max(0.0, (box[0] + box[2] / 2.0)), 0.9999) * image.shape[1])
                    top = int(min(max(0.0, (box[1] - box[3] / 2.0)), 0.9999) * image.shape[0])
                    bottom = int(min(max(0.0, (box[1] + box[3] / 2.0)), 0.9999) * image.shape[0])
                    cv2.rectangle(image, (left, top), (right, bottom), (238, 130, 238), 2)
            
            plt.imsave(output_path, image)
            
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

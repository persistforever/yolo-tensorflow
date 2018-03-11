# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/01/08
# description: model managering in table detection
from __future__ import print_function
import sys
import os
import time
import math
import numpy
import random
import matplotlib.pyplot as plt
from ctypes import c_double, cast, POINTER
import cv2
import tensorflow as tf
from pdfinsight.ai.yolo_tf.src.layer.conv_layer import ConvLayer
from pdfinsight.ai.yolo_tf.src.layer.pool_layer import PoolLayer
import pdfinsight.ai.yolo_tf.src.tools.utils as utils


class Model():
    
    def __init__(self, 
        n_channel, 
        max_objects,
        image_x_size, 
        image_y_size, 
        cell_x_size, 
        cell_y_size, 
        box_per_cell, 
        batch_size, 
        buffer_size, 
        is_valid=False, 
        update_function='momentum', 
        learning_rate=0.01,
        is_lr_decay=False):

        # 设置参数
        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.n_channel = n_channel
        self.max_objects = max_objects
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.n_boxes = box_per_cell
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.is_valid = is_valid
        self.update_function = update_function
        self.learning_rate = learning_rate
        self.is_lr_decay = is_lr_decay
        self.n_coord = 8 if self.is_duplex else 4
        
        self.index_size = (self.batch_size)
        self.image_size = (self.batch_size, self.image_y_size, self.image_x_size, self.n_channel)
        self.coord_true_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects, 8)
        self.object_mask_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects)
        self.unpos_coord_true_size = (self.batch_size, self.max_objects, 8)
        self.unpos_object_mask_size = (self.batch_size, self.max_objects)
        self.object_nums_size = (self.batch_size, self.cell_y_size, self.cell_x_size)
        
        self.place_holders = []
        for i in range(self.n_gpus):
            # 输入变量
            self.images = tf.placeholder(
                dtype=tf.int32, 
                shape=[int(self.batch_size/self.n_gpus), self.image_y_size, self.image_x_size, 1], 
                name='images')
            self.coord_true = tf.placeholder(
                dtype=tf.float32, 
                shape=[int(self.batch_size/self.n_gpus), self.cell_y_size, self.cell_x_size, self.max_objects, 8], 
                name='coord_true')
            self.object_mask = tf.placeholder(
                dtype=tf.float32, 
                shape=[int(self.batch_size/self.n_gpus), self.cell_y_size, self.cell_x_size, self.max_objects], 
                name='object_mask')
            self.unpos_coord_true = tf.placeholder(
                dtype=tf.float32,
                shape=[int(self.batch_size/self.n_gpus), self.max_objects, 8],
                name='unpos_coord_true')
            self.unpos_object_mask = tf.placeholder(
                dtype=tf.float32,
                shape=[int(self.batch_size/self.n_gpus), self.max_objects],
                name='unpos_object_mask')
            self.object_nums = tf.placeholder(
                dtype=tf.int32,
                shape=[int(self.batch_size/self.n_gpus), self.cell_y_size, self.cell_x_size],
                name='object_nums')
            
            self.place_holders.append({
                'images': self.images, 'coord_true': self.coord_true, 
                'object_mask': self.object_mask, 'unpos_coord_true': self.unpos_coord_true, 
                'unpos_object_mask': self.unpos_object_mask, 'object_nums': self.object_nums})
        self.global_step = tf.Variable(0, dtype=tf.float32, name='global_step')

    def train_init(self, network, backup_dir):
        time.sleep(5)

        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True))
        
        if self.update_function == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.9)
        elif self.update_function == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.update_function == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

        # 构建模型和优化器
        self.network = network
        # 先计算loss
        with tf.name_scope('cal_loss_and_eval'):
            self.avg_loss, self.coord_loss, self.noobject_loss, self.object_loss, \
                self.weight_decay_loss, self.iou_value, self.object_value, self.noobject_value = \
                    self.network.get_loss(
                        self.place_holders[0]['images'],
                        self.place_holders[0]['coord_true'], 
                        self.place_holders[0]['object_mask'],
                        self.place_holders[0]['unpos_coord_true'],
                        self.place_holders[0]['unpos_object_mask'],
                        self.place_holders[0]['object_nums'],
                        self.global_step, 'gpu0')

        # 然后求误差并更新参数
        with tf.name_scope('optimize'):
            self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                global_step=self.global_step)
            
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=500)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        self.valid_logits = self.network.get_inference(self.place_holders[0]['images'], 
        
    def train(self, processor, network, backup_dir, logs_dir, n_iters=500000):
        sub_dir = 'adaption' if self.is_adaption else 'basic'
        self.train_init(network, backup_dir)
        
        # 训练开始前保存1次模型
        model_path = os.path.join(backup_dir, sub_dir, 'model_0.ckpt')
        self.saver.save(self.sess, model_path)
                
        # 模型训练
        process_images = 0
        
        start_time = time.time()
        data_spend, model_spend, max_valid_value, max_train_value = 0.0, 0.0, 0.0, 0.0
        
        print('\nstart training ...\n')
        for n_iter in range(self.n_restart, n_iters+4):
            # 获取数据
            st = time.time()
            data = processor.shared_memory.get()
            
            # 将shared_memory中的数据取出
            accum_size = 0
            
            batch_image_indexs = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.index_size)], self.index_size)
            accum_size += numpy.prod(self.index_size)
            
            batch_images = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.image_size)], self.image_size)
            accum_size += numpy.prod(self.image_size)
            
            batch_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.coord_true_size)], self.coord_true_size)
            accum_size += numpy.prod(self.coord_true_size)
            
            batch_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.object_mask_size)], self.object_mask_size)
            accum_size += numpy.prod(self.object_mask_size)

            batch_unpos_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.unpos_coord_true_size)], 
                self.unpos_coord_true_size)
            accum_size += numpy.prod(self.unpos_coord_true_size)

            batch_unpos_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.unpos_object_mask_size)],
                self.unpos_object_mask_size)
            accum_size += numpy.prod(self.unpos_object_mask_size)

            batch_object_nums = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.object_nums_size)], self.object_nums_size)
            accum_size += numpy.prod(self.object_nums_size)
            
            et = time.time()
            data_time = et - st

            st = time.time()
            sub_batch_size = int(self.batch_size / self.n_gpus)
            feed_dict = {}
            
            # 生成feed_dict
            for i in range(self.n_gpus):
                feed_dict[self.place_holders[i]['images']] = \
                    batch_fore_images[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:]
                
                feed_dict[self.place_holders[i]['coord_true']] = \
                    batch_coord_true[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:,:]
                
                feed_dict[self.place_holders[i]['object_mask']] = \
                    batch_object_mask[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:]
                
                feed_dict[self.place_holders[i]['unpos_coord_true']] = \
                    batch_unpos_coord_true[i*sub_batch_size:(i+1)*sub_batch_size,:,:]

                feed_dict[self.place_holders[i]['unpos_object_mask']] = \
                    batch_unpos_object_mask[i*sub_batch_size:(i+1)*sub_batch_size,:]

                feed_dict[self.place_holders[i]['object_nums']] = \
                    batch_object_nums[i*sub_batch_size:(i+1)*sub_batch_size,:,:]
                
            et = time.time()
            feed_time = et - st
            
            st = time.time()
            [_, avg_loss, coord_loss, noobject_loss, object_loss, weight_decay_loss, \
                iou_value, object_value, noobject_value] = self.sess.run(
                    fetches=[
                        self.optimizer_handle, self.avg_loss, self.coord_loss, 
                        self.noobject_loss, self.object_loss, self.weight_decay_loss,
                        self.iou_value, self.object_value, self.noobject_value], 
                    feed_dict=feed_dict)
            et = time.time()
            model_time = et - st
           
            process_images += self.batch_size
            
            end_time = time.time()
            spend = (end_time - start_time) / 3600.0
            
            print('[%d] data time: %.4fs, model time: %.4fs, spend: %.4fh, image_nums: %d' % (
                n_iter, data_time, model_time, spend, process_images))

            # 每1轮训练观测一次train_loss    
            print('[%d] train loss: %.6f, coord loss: %.6f, noobject loss: %.6f, '
                'object loss: %.6f, weight loss: %.6f, recon loss: %.6f' % (
                n_iter, avg_loss, coord_loss, noobject_loss, object_loss, 
                weight_decay_loss, recon_loss))
            sys.stdout.flush()
            
            # 每1轮观测一次训练集evaluation
            print('[%d] inner IOU: %.6f, outer IOU: %.6f, '
                'object: %.6f, noobject: %.6f, overlap: %.6f\n' % (
                n_iter, iou_value, outer_iou_value, object_value, noobject_value, overlap_value))
            sys.stdout.flush()

            # 每固定轮数验证一次模型
            valid_freq = 1000
            need_valid = False
            for t in range(-3,4,1):
                if (n_iter+t) % valid_freq == 0 and n_iter >= 10:
                    need_valid = True
                    break
            
            if need_valid and self.is_valid:
                # 观测一次basic验证集evaluation
                precision_array, recall_array, f1_array, overlap = self.valid_model(
                    processor, model_path, logs_dir, mode='valid_basic')
                print('[%d] valid '
                    'p@0.4: %.6f, r@0.4: %.6f, f1@0.4: %.6f, '
                    'p@0.5: %.6f, r@0.5: %.6f, f1@0.5: %.6f, '
                    'p@0.6: %.6f, r@0.6: %.6f, f1@0.6: %.6f, '
                    'p@0.7: %.6f, r@0.7: %.6f, f1@0.7: %.6f, '
                    'p@0.8: %.6f, r@0.8: %.6f, f1@0.8: %.6f, '
                    'p@0.9: %.6f, r@0.9: %.6f, f1@0.9: %.6f, '
                    'tp@1.0: %.6f, tr@1.0: %.6f, tf1@1.0: %.6f, '
                    'wp@1.0: %.6f, wr@1.0: %.6f, wf1@1.0: %.6f, '
                    'overlap: %.6f\n' % (
                    n_iter, precision_array[0], recall_array[0], f1_array[0], 
                    precision_array[1], recall_array[1], f1_array[1],
                    precision_array[2], recall_array[2], f1_array[2],
                    precision_array[3], recall_array[3], f1_array[3],
                    precision_array[4], recall_array[4], f1_array[4],
                    precision_array[5], recall_array[5], f1_array[5],
                    precision_array[6], recall_array[6], f1_array[6],
                    precision_array[7], recall_array[7], f1_array[7],
                    overlap))
                valid_value = f1_array[6]

                if train_value >= max_train_value:
                    max_train_value = train_value
                    print('update best train textf1@1.0: %.4f\n' % (max_train_value))
            
            # 每固定轮数保存一次模型
            if n_iter % 5000 == 0:
                model_path = os.path.join(backup_dir, sub_dir, 'model_%d.ckpt' % (n_iter))
                self.saver.save(self.sess, model_path)
            
            sys.stdout.flush()
        
        self.sess.close()

    def valid_init(self, processor, network):
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True))
        self.network = network
        self.valid_logits = self.network.get_inference(
            self.place_holders[0]['fore_images'], 
            self.place_holders[0]['line_images'],
            self.place_holders[0]['back_images'])
        
    def valid_all_models(self, processor, network, backup_dir, logs_dir, n_iters=100000):
        sub_dir = 'adaption' if self.is_adaption else 'basic'
        
        # 验证backup_dir中的每一个模型
        for n_iter in range(n_iters):
            if (n_iter <= 1000 and n_iter % 200 == 0) or (1000 < n_iter <= 10000 and n_iter % 2000 == 0) \
                or (n_iter > 10000 and n_iter % 20000 == 0):
                model_path = os.path.join(backup_dir, sub_dir, 'model_%d.ckpt' % (n_iter))
                # 读取模型
                self.valid_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
                assert(os.path.exists(model_path+'.index'))
                self.valid_saver.restore(self.sess, model_path)
                print('read model from %s' % (model_path))
                
                precision_array, recall_array, f1_array, overlap = self.valid_model(
                    processor, model_path, logs_dir, mode='valid')
                print('[%d] p@0.4: %.6f, r@0.4: %.6f, f1@0.4: %.6f, '
                    'p@0.5: %.6f, r@0.5: %.6f, f1@0.5: %.6f, '
                    'p@0.6: %.6f, r@0.6: %.6f, f1@0.6: %.6f, '
                    'p@0.7: %.6f, r@0.7: %.6f, f1@0.7: %.6f, '
                    'p@0.8: %.6f, r@0.8: %.6f, f1@0.8: %.6f, '
                    'p@0.9: %.6f, r@0.9: %.6f, f1@0.9: %.6f, '
                    'p@1.0: %.6f, r@1.0: %.6f, f1@1.0: %.6f, '
                    'overlap: %.6f\n' % (
                    n_iter, precision_array[0], recall_array[0], f1_array[0], 
                    precision_array[1], recall_array[1], f1_array[1],
                    precision_array[2], recall_array[2], f1_array[2],
                    precision_array[3], recall_array[3], f1_array[3],
                    precision_array[4], recall_array[4], f1_array[4],
                    precision_array[5], recall_array[5], f1_array[5],
                    precision_array[6], recall_array[6], f1_array[6],
                    overlap))
                sys.stdout.flush()

    def valid_model(self, processor, model_path, output_dir, mode='valid'):
        n_ious = 8
        preds_denominator = 0
        trues_denominator = 0
        right_array = [0.0] * n_ious
        overlap_numerator = 0
        overlap_denominator = 0
        self.batch_size = int(self.batch_size / self.n_gpus)

        for i in range(int(processor.n_valid / self.batch_size) - 1):
            # 获取数据并进行数据增强
            batch_images, batch_datasets = processor.dataset_producer(
                mode=mode, indexs=[i*self.batch_size+t for t in range(self.batch_size)])
            batch_images = numpy.zeros((self.batch_size, self.image_y_size, self.image_x_size, 3))
            
            [logits] = self.sess.run(
                fetches=[self.valid_logits],
                feed_dict={self.place_holders[0]['images']: batch_images})
            
            # 获得预测的框
            preds_boxes = self.get_pred_boxes(logits, batch_datasets, self.batch_size)
            for boxes in preds_boxes:
                preds_denominator += len(boxes)

            # 获得真实的框
            trues_boxes = self.get_true_boxes(batch_datasets, self.batch_size)
            for boxes in trues_boxes:
                trues_denominator += len(boxes)

            # 获得预测的框和真实的框的对应pair
            pairs_boxes_text = self.get_pair_boxes(trues_boxes_text, preds_boxes_text)
            pairs_boxes_word = self.get_pair_boxes(trues_boxes_word, preds_boxes_word)

            if mode == 'valid_basic' and i == 0 and self.is_observe:
                self.write_valid_images(
                    batch_datasets, pairs_boxes_text, preds_boxes_text, trues_boxes_text, output_dir)

            # 计算每个真实框对应的IOU最大的预测框
            for j in range(self.batch_size):
                docid = batch_datasets[j]['docid']
                pageid = batch_datasets[j]['pageid']
                texts = batch_datasets[j]['content']['processed_texts']
                shape = batch_datasets[j]['content']['resized_size']

                for p, t, best_iou in pairs_boxes_text[j]:
                    if self.judge_pred_true_matched(preds_boxes_text[j][p], trues_boxes_text[j][t], texts):
                        right_array[6] += 1
                    if best_iou >= 0.9:
                        right_array[5] += 1
                    if best_iou >= 0.8:
                        right_array[4] += 1
                    if best_iou >= 0.7:
                        right_array[3] += 1
                    if best_iou >= 0.6:
                        right_array[2] += 1
                    if best_iou >= 0.5:
                        right_array[1] += 1
                    if best_iou >= 0.4:
                        right_array[0] += 1
                
                for p, t, best_iou in pairs_boxes_word[j]:
                    if self.judge_pred_true_matched(preds_boxes_text[j][p], trues_boxes_text[j][t], 
                        texts, is_text=False):
                        right_array[7] += 1

            # 计算每两个预测框之间的overlap面积
            overlap_one_numerator = 0
            overlap_one_denominator = 0
            for j in range(self.batch_size):
                for a in range(len(preds_boxes_text[j])):
                    for b in range(a+1, len(preds_boxes_text[j])):
                        iou = self.calculate_iou_py(preds_boxes_text[j][a], preds_boxes_text[j][b], mode='ltrb')
                        overlap_one_numerator += iou
                        overlap_one_denominator += 1
            overlap = 1.0 * overlap_one_numerator / overlap_one_denominator \
                if overlap_one_denominator != 0 else 0.0
            overlap_numerator += overlap
            overlap_denominator += 1

        self.batch_size = int(self.batch_size * self.n_gpus)
        precision_array = [0.0] * n_ious
        recall_array = [0.0] * n_ious
        f1_array = [0.0] * n_ious
        for i in range(n_ious):
            precision_array[i] = 1.0 * right_array[i] / preds_denominator \
                if preds_denominator != 0 else 0.0
            recall_array[i] = 1.0 * right_array[i] / trues_denominator \
                if trues_denominator != 0 else 0.0
            f1_array[i] = 2 * precision_array[i] * recall_array[i] / (
                precision_array[i] + recall_array[i]) if precision_array[i] != 0 or \
                recall_array[i] !=0 else 0.0
        overlap = 1.0 * overlap_numerator / overlap_denominator \
            if overlap_denominator != 0 else 0.0

        return precision_array, recall_array, f1_array, overlap
    
    def test_model(self, processor, network, model_path, output_dir):
        self.deploy_init(processor, network, model_path)
        
        if not os.path.exists(os.path.join(output_dir, 'predictions')):
            os.mkdir(os.path.join(output_dir, 'predictions'))
        for i in range(int(processor.n_test_basic / self.batch_size)-1):
            # 获取数据并进行数据增强
            batch_images, batch_datasets = processor.dataset_producer(
                mode='test_basic', indexs=[i*self.batch_size+t for t in range(self.batch_size)])
            
            [logits] = self.sess.run(
                fetches=[self.deploy_logits], 
                feed_dict={self.images: batch_images})
            
            # 获得预测的框
            preds_boxes = self.get_pred_boxes(logits, batch_datasets, self.batch_size, is_text=True)
            
            for j in range(self.batch_size):
                docid = batch_datasets[j]['docid']
                pageid = int(batch_datasets[j]['pageid'])
                output_path = os.path.join(output_dir, 'predictions', '%s_%d.png' % (
                    docid, pageid))
                image_path = batch_datasets[j]['path']
                show_path = batch_datasets[j]['content']['orig_image_path']
                print(show_path)
                image = cv2.imread(show_path)
                
                # 画预测的框
                for box in preds_boxes[j]:
                    [left, top, right, bottom] = [int(t) for t in box]
                    cv2.rectangle(image, (left, top), (right, bottom), (238, 192, 126), 2) # blue
                
                cv2.imwrite(output_path, image)
        self.test_sess.close()
        print('Test Finish!')
    
    def get_pred_boxes(self, logits, batch_datasets, batch_size, is_text=True):
        box_preds = numpy.reshape(logits, (
            batch_size, self.cell_y_size, self.cell_x_size, 1+self.n_coord))
        new_box_preds = self.get_direct_position_py(box_preds[:,:,:,1:5])
       
        pred_boxes = []
        for j in range(batch_size):
            shape = batch_datasets[j]['content']['resized_size']

            # 获得预测的preds
            preds = []
            for x in range(self.cell_x_size):
                for y in range(self.cell_y_size):
                    prob = box_preds[j,y,x,0]
                    box = new_box_preds[j,y,x,0:4]
                    if prob >= self.network.pred_thresh:
                        preds.append([box, prob])
            
            # 排序并去除多余的box
            preds = sorted(preds, key=lambda x: x[1], reverse=True)
            for x in range(len(preds)):
                if preds[x][1] < self.network.pred_thresh:
                    continue
                for y in range(x+1, len(preds)):
                    iou = self.calculate_iou_py(preds[x][0], preds[y][0], mode='xywh')
                    if iou > self.network.nms_thresh:
                        preds[y][1] = 0.0
            
            # 画预测的框
            boxes = []
            for k in range(len(preds)):
                if preds[k][1] >= self.network.pred_thresh:
                    [x, y, w, h] = preds[k][0]
                    left = int(round(min(max(0.0, x - w / 2.0), 0.9999) * self.image_x_size))
                    top = int(round(min(max(0.0, y - h / 2.0), 0.9999) * self.image_y_size))
                    right = int(round(min(max(0.0, x + w / 2.0), 0.9999) * self.image_x_size))
                    bottom = int(round(min(max(0.0, y + h / 2.0), 0.9999) * self.image_y_size))

                    boxes.append([left, top, right, bottom, preds[k][1]])
            
            pred_boxes.append(boxes)
        
        return pred_boxes
    
    def get_true_boxes(self, batch_datasets, batch_size, is_text=True):
        trues_boxes = []
        
        for j in range(batch_size):
            tables = batch_datasets[j]['labels']

            true_boxes = []
            for table in tables:
                [x, y, w, h] = table
                left = int(round(min(max(0.0, x - w / 2.0), 0.9999) * self.image_x_size))
                top = int(round(min(max(0.0, y - h / 2.0), 0.9999) * self.image_y_size))
                right = int(round(min(max(0.0, x + w / 2.0), 0.9999) * self.image_x_size))
                bottom = int(round(min(max(0.0, y + h / 2.0), 0.9999) * self.image_y_size))
                
                true_boxes.append([left, top, right, bottom])

            trues_boxes.append(true_boxes)

        return trues_boxes

    def get_pair_boxes(self, trues_boxes, preds_boxes):
        """
        获取每个预测框对应的真实框的pair对
        """
        pairs_boxes = []
        for i in range(self.batch_size):
            pred_pair_dict = {}
            pair_boxes = []
            for t in range(len(trues_boxes[i])):
                best_n, best_iou = -1, 0.4
                for p in range(len(preds_boxes[i])):
                    iou = self.calculate_iou_py(preds_boxes[i][p], trues_boxes[i][t], mode='ltrb')
                    if iou >= best_iou:
                        best_iou = iou
                        best_n = p
                if best_n != -1:
                    if best_n not in pred_pair_dict:
                        pred_pair_dict[best_n] = []
                    pred_pair_dict[best_n].append([t, best_iou])
            for p in pred_pair_dict:
                [best_n, best_iou] = max(pred_pair_dict[p], key=lambda x: x[1])
                pair_boxes.append([p, best_n, best_iou])
            pairs_boxes.append(pair_boxes)

        return pairs_boxes

    def get_direct_position_py(self, coord_pred):
        # 计算bx
        offset_x = numpy.reshape(range(0, self.cell_x_size), newshape=(1, 1, self.cell_x_size, 1))
        offset_x = numpy.tile(offset_x, (self.batch_size, self.cell_y_size, 1, 1))
        offset_x = numpy.array(offset_x, dtype='float')
        x_pred = (coord_pred[:,:,:,0:1] + offset_x) / self.cell_x_size
        
        # 计算by
        offset_y = numpy.reshape(range(0, self.cell_y_size), newshape=(1, self.cell_y_size, 1, 1))
        offset_y = numpy.tile(offset_y, (self.batch_size, 1, self.cell_x_size, 1))
        offset_y = numpy.array(offset_y, dtype='float')
        y_pred = (coord_pred[:,:,:,1:2] + offset_y) / self.cell_y_size
        
        new_coord_pred = numpy.concatenate([x_pred, y_pred, coord_pred[:,:,:,2:4]], axis=3)
        
        return new_coord_pred
    
    def calculate_iou_py(self, box_pred, box_label, mode='xywh'):
        if mode == 'xywh':
            box1 = [box_pred[0] - box_pred[2] / 2.0, box_pred[1] - box_pred[3] / 2.0,
                box_pred[0] + box_pred[2] / 2.0, box_pred[1] + box_pred[3] / 2.0]
            box2 = [box_label[0] - box_label[2] / 2.0, box_label[1] - box_label[3] / 2.0,
                box_label[0] + box_label[2] / 2.0, box_label[1] + box_label[3] / 2.0]
        elif mode == 'ltrb':
            box1 = box_pred
            box2 = box_label
        left = max(box1[0], box2[0])
        top = max(box1[1], box2[1])
        right = min(box1[2], box2[2])
        bottom = min(box1[3], box2[3])
        if right <= left or bottom <= top:
            iou = 0.0
        else:
            inter_area = (right - left) * (bottom - top)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)
        
        return iou

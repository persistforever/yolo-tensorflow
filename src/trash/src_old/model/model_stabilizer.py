# -*- coding: utf8 -*-
# author: ronniecao
# time: 2017/12/19
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


class Model():
    
    def __init__(self, n_channel, image_x_size, image_y_size, max_objects_per_image,
        cell_x_size, cell_y_size, box_per_cell, batch_size, buffer_size, n_gpus=1,
        is_multigpu=False, is_origin_matrix=False, is_valid=False, 
        is_underlap=False, is_duplex=False):

        # 设置参数
        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.n_channel = n_channel
        self.max_objects = max_objects_per_image
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.n_boxes = box_per_cell
        self.batch_size = batch_size
        self.n_gpus = n_gpus
        self.is_multigpu = is_multigpu
        self.buffer_size = buffer_size
        self.is_origin_matrix = is_origin_matrix
        self.is_valid = is_valid
        self.is_underlap = is_underlap
        self.is_duplex = is_duplex
        self.n_coord = 8 if self.is_duplex else 4
        self.image_index_size = (self.batch_size)
        if self.is_origin_matrix:
            self.image_all_size = (self.batch_size, self.image_y_size, self.image_x_size, 1)
        else:
            self.image_all_size = (self.batch_size, self.image_y_size, self.image_x_size, self.n_channel)
        self.coord_true_size = (self.batch_size, self.cell_y_size, self.cell_x_size, 
            self.max_objects, self.n_coord)
        self.coord_mask_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects)
        self.up_coord_true_size = (self.batch_size, self.max_objects, self.n_coord)
        self.up_coord_mask_size = (self.batch_size, self.max_objects)
        self.dataset_size = sum([numpy.prod(t) for t in [
            self.image_all_size, self.coord_true_size, self.coord_mask_size, 
            self.up_coord_true_size, self.up_coord_mask_size]])
        
        self.place_holders = []
        for i in range(self.n_gpus):
            # 输入变量
            if self.is_origin_matrix:
                self.images = tf.placeholder(
                    dtype=tf.int32, shape=[
                        int(self.batch_size/self.n_gpus), self.image_y_size, self.image_x_size, 1], 
                    name='images')
            else:
                self.images = tf.placeholder(
                    dtype=tf.float32, shape=[
                        int(self.batch_size/self.n_gpus), self.image_y_size, self.image_x_size, self.n_channel], 
                    name='images')
            self.coord_true = tf.placeholder(
                dtype=tf.float32, shape=[
                    int(self.batch_size/self.n_gpus), self.cell_y_size, self.cell_x_size, self.max_objects, self.n_coord], 
                name='coord_true')
            self.object_mask = tf.placeholder(
                dtype=tf.float32, shape=[
                    int(self.batch_size/self.n_gpus), self.cell_y_size, self.cell_x_size, self.max_objects], 
                name='object_mask')
            self.unpositioned_coord_true = tf.placeholder(
                dtype=tf.float32, shape=[
                    int(self.batch_size/self.n_gpus), self.max_objects, self.n_coord], 
                name='unpositioned_coord_true')
            self.unpositioned_object_mask = tf.placeholder(
                dtype=tf.float32, shape=[
                    int(self.batch_size/self.n_gpus), self.max_objects], 
                name='unpositioned_object_mask')
            self.place_holders.append({
                'images': self.images, 'coord_true': self.coord_true, 'object_mask': self.object_mask,
                'unpositioned_coord_true': self.unpositioned_coord_true, 
                'unpositioned_object_mask': self.unpositioned_object_mask})
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step')
        
    def train(self, processor, network, backup_dir, n_iters=500000, batch_size=128):
        time.sleep(15)
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True))
        
        # 构建模型和优化器
        self.network = network
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)

        if not self.is_multigpu:
            # 先计算loss
            with tf.name_scope('cal_loss_and_eval'):
                self.avg_loss, self.iou_value, self.object_value, self.noobject_value, \
                    self.recall_value, self.overlap_value, self.outer_iou_value = \
                        self.network.get_loss(
                            self.place_holders[0]['images'], 
                            self.place_holders[0]['coord_true'], 
                            self.place_holders[0]['object_mask'], 
                            self.place_holders[0]['unpositioned_coord_true'], 
                            self.place_holders[0]['unpositioned_object_mask'], 
                            self.global_step, 'gpu0')
    
            # 然后求误差并更新参数
            with tf.name_scope('optimize'):
                self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                    global_step=self.global_step)
        else:
            tower_grads, tower_evals = [], []
            # 先在每个GPU上计算loss
            for i in range(self.n_gpus):
                with tf.device('/gpu:%d' % (i)):
                    with tf.name_scope('cal_loss_and_eval_%d' % (i)) as scope:
                        evals = self.network.get_loss(
                            self.place_holders[i]['images'], 
                            self.place_holders[i]['coord_true'], 
                            self.place_holders[i]['object_mask'], 
                            self.place_holders[i]['unpositioned_coord_true'], 
                            self.place_holders[i]['unpositioned_object_mask'], 
                            self.global_step, 'gpu%d' % (i))
                        grads_and_vars = self.optimizer.compute_gradients(evals[0])
                        tower_grads.append(grads_and_vars)
                        tower_evals.append(evals)
            
            with tf.device('/cpu:0'):
                with tf.name_scope('optimize'):
                    avg_grads_and_vars, avg_evals = self.average_gradients(tower_grads, tower_evals)
                    self.avg_loss, self.iou_value, self.object_value, self.noobject_value, \
                        self.recall_value, self.overlap_value, self.outer_iou_value = avg_evals
                    self.optimizer_handle = self.optimizer.apply_gradients(
                        avg_grads_and_vars, global_step=self.global_step)
        
        # 模型保存器
        self.saver = tf.train.Saver(
            var_list=tf.global_variables(), write_version=tf.train.SaverDef.V2, max_to_keep=50)
        # 模型初始化
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, os.path.join(backup_dir, 'model_start.ckpt'))
                
        # 模型训练
        process_images = 0
        
        start_time = time.time()
        data_spend, model_spend = 0.0, 0.0
        for n_iter in range(0, n_iters):
            # 获取数据
            st = time.time()
            data = processor.shared_memory.get()
            
            accum_size = 0
            batch_image_indexs = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.image_index_size)], self.image_index_size)
            accum_size += numpy.prod(self.image_index_size)
            batch_images = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.image_all_size)], self.image_all_size)
            accum_size += numpy.prod(self.image_all_size)
            batch_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.coord_true_size)], self.coord_true_size)
            accum_size += numpy.prod(self.coord_true_size)
            batch_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.coord_mask_size)], self.coord_mask_size)
            accum_size += numpy.prod(self.coord_mask_size)
            batch_unpositioned_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.up_coord_true_size)], self.up_coord_true_size)
            accum_size += numpy.prod(self.up_coord_true_size)
            batch_unpositioned_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.up_coord_mask_size)], self.up_coord_mask_size)
            et = time.time()
            data_time = et - st

            st = time.time()
            sub_batch_size = int(self.batch_size / self.n_gpus)
            feed_dict = {}
            for i in range(self.n_gpus):
                feed_dict[self.place_holders[i]['images']] = \
                    batch_images[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:]
                feed_dict[self.place_holders[i]['coord_true']] = \
                    batch_coord_true[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:,:]
                feed_dict[self.place_holders[i]['object_mask']] = \
                    batch_object_mask[i*sub_batch_size:(i+1)*sub_batch_size,:,:,:]
                feed_dict[self.place_holders[i]['unpositioned_coord_true']] = \
                    batch_unpositioned_coord_true[i*sub_batch_size:(i+1)*sub_batch_size,:,:]
                feed_dict[self.place_holders[i]['unpositioned_object_mask']] = \
                    batch_unpositioned_object_mask[i*sub_batch_size:(i+1)*sub_batch_size,:]
            et = time.time()
            feed_time = et - st
            
            st = time.time()
            [_, avg_loss, iou_value, object_value, noobject_value, recall_value, overlap_value,
                outer_iou_value] = \
                self.sess.run(
                    fetches=[
                        self.optimizer_handle, self.avg_loss, self.iou_value, 
                        self.object_value, self.noobject_value, self.recall_value, 
                        self.overlap_value, self.outer_iou_value], 
                    feed_dict=feed_dict)
            et = time.time()
            model_time = et - st
           
            print('[%d] data time: %.4f, feed time: %.4f, model time: %.4f' % (
                n_iter, data_time, feed_time, model_time))

            process_images += batch_size
            
            end_time = time.time()
            spend = end_time - start_time
            
            # 每1轮训练观测一次train_loss    
            print('[%d] train_loss: %.4f, image_nums: %d, spend: %.2f' % (
                n_iter, avg_loss, process_images, spend))
            sys.stdout.flush()
            
            # 每1轮观测一次训练集evaluation
            print('[%d] IOU: %.6f, Object: %.6f, Noobject: %.6f, OverlapLoss: %.6f, Overlap: %.6f, '
                'OuterIOU: %.6f' % (
                n_iter, iou_value, object_value, noobject_value, recall_value, overlap_value,
                outer_iou_value))
            sys.stdout.flush()
            
            # 每固定轮数保存一次模型
            if n_iter % 1000 == 0:
                model_path = os.path.join(backup_dir, 'model_%d.ckpt' % (n_iter))
                self.saver.save(self.sess, model_path)
                # 每保存一次模型，观测一次验证集evaluation
                if self.is_valid:
                    nega_error_rate, posi_error_rate, precision, overlap = self.valid_model(
                        processor, network, model_path)
                    print('[%d] negative error: %.6f, positive error: %.6f, precision: %.6f, '
                        'overlap: %.6f\n' % (
                        n_iter, nega_error_rate, posi_error_rate, precision, overlap))
                    sys.stdout.flush()

            print()
            sys.stdout.flush()
        
        self.sess.close()

    def valid_all_models(self, processor, network, backup_dir, n_iters=100000):
        # 验证backup_dir中的每一个模型
        for n_iter in range(n_iters):
            if (n_iter <= 1000 and n_iter % 200 == 0) or (1000 < n_iter <= 10000 and n_iter % 2000 == 0) \
                or (n_iter > 10000 and n_iter % 20000 == 0):
                model_path = os.path.join(backup_dir, 'model_%d.ckpt' % (n_iter))
                nega_error_rate, posi_error_rate, precision, overlap = self.valid_model(
                    processor, network, model_path)
                print('[%d] negative error: %.6f, positive error: %.6f, '
                    'precision: %.6f, overlap: %.6f\n' % (
                    n_iter, nega_error_rate, posi_error_rate, precision, overlap))
                sys.stdout.flush()

    def valid_model(self, processor, network, model_path):
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.valid_sess = tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True))
        
        # 读取模型
        self.valid_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        assert(os.path.exists(model_path+'.index'))
        self.valid_saver.restore(self.valid_sess, model_path)
        print('read model from %s' % (model_path))
        
        self.network = network
        self.logits = self.network.get_inference(self.place_holders[0]['images'])
        
        nega_error_numerator = 0
        nega_error_denominator = 0
        posi_error_numerator = 0
        posi_error_denominator = 0
        precision_numerator = 0
        precision_denominator = 0
        overlap_numerator = 0
        overlap_denominator = 0
        self.batch_size = int(self.batch_size / self.n_gpus)
        for i in range(int(processor.n_valid / self.batch_size) - 1):
            # 获取数据并进行数据增强
            batch_images = processor.dataset_producer(
                mode='valid', index=i*self.batch_size, batch_size=self.batch_size)
            
            [logits] = self.valid_sess.run(
                fetches=[self.logits],
                feed_dict={self.place_holders[0]['images']: batch_images})
            
            # 获得这一批的原始数据
            batch_datasets = processor.validsets[i*self.batch_size: (i+1)*self.batch_size]

            # 获得预测的框
            pred_boxes = self.get_pred_boxes(
                logits, batch_datasets, self.batch_size, is_origin_size=True)
            for boxes in pred_boxes:
                nega_error_denominator += len(boxes)

            # 获得真实的框
            true_boxes = self.get_true_boxes(
                batch_datasets, self.batch_size)
            for boxes in true_boxes:
                posi_error_denominator += len(boxes)

            # 计算每个真实框对应的IOU最大的预测框
            for j in range(self.batch_size):
                docid = batch_datasets[j]['docid']
                pageid = batch_datasets[j]['pageid']
                texts = batch_datasets[j]['content']['processed_texts']
                shape = batch_datasets[j]['content']['size']
                
                matched_pred_boxes = []
                for true_box in true_boxes[j]:
                    best_iou, best_n = 0.4, -1
                    for k in range(len(pred_boxes[j])):
                        iou = self.calculate_iou_py(pred_boxes[j][k], true_box, mode='ltrb')
                        if iou > best_iou:
                            best_iou = iou
                            best_n = k
                    if best_n == -1:
                        posi_error_numerator += 1
                    else:
                        matched_pred_boxes.append(best_n)
                        precision_denominator += 1
                        if self.judge_pred_true_matched(pred_boxes[j][best_n], true_box, texts):
                            precision_numerator += 1
                nega_error_denominator += len(pred_boxes[j])
                nega_error_numerator += len(pred_boxes[j]) - len(matched_pred_boxes)

            # 计算每两个预测框之间的overlap面积
            overlap_one_numerator = 0
            overlap_one_denominator = 0
            for j in range(self.batch_size):
                for a in range(len(pred_boxes[j])):
                    for b in range(a+1, len(pred_boxes[j])):
                        iou = self.calculate_iou_py(pred_boxes[j][a], pred_boxes[j][b], mode='ltrb')
                        overlap_one_numerator += iou
                        overlap_one_denominator += 1
            overlap = 1.0 * overlap_one_numerator / overlap_one_denominator \
                if overlap_one_denominator != 0 else 0.0
            overlap_numerator += overlap
            overlap_denominator += 1

        self.valid_sess.close()
        self.batch_size = int(self.batch_size * self.n_gpus)
        nega_error_rate = 1.0 * nega_error_numerator / nega_error_denominator \
            if nega_error_denominator != 0 else 0.0
        posi_error_rate = 1.0 * posi_error_numerator / posi_error_denominator \
            if posi_error_denominator != 0 else 0.0
        precision = 1.0 * precision_numerator / precision_denominator \
            if precision_denominator != 0 else 0.0
        overlap = 1.0 * overlap_numerator / overlap_denominator \
            if overlap_denominator != 0 else 0.0

        return nega_error_rate, posi_error_rate, precision, overlap
    
    def test_model(self, processor, network, model_path, output_dir):
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.test_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.test_sess, model_path)
        print('read model from %s' % (model_path))
        
        self.network = network
        self.logits = self.network.get_inference(self.images)
        
        if not os.path.exists(os.path.join(output_dir, 'predictions')):
            os.mkdir(os.path.join(output_dir, 'predictions'))
        for i in range(int(processor.n_test / self.batch_size)-1):
            # 获取数据并进行数据增强
            batch_images = processor.dataset_producer(
                mode='test', index=i*self.batch_size, batch_size=self.batch_size)
            
            [logits] = self.test_sess.run(
                fetches=[self.logits], 
                feed_dict={self.images: batch_images})
            
            # 获得这一批的原始数据
            batch_datasets = processor.testsets[i*self.batch_size: (i+1)*self.batch_size]

            # 获得预测的框
            pred_boxes = self.get_pred_boxes(logits, batch_datasets, 
                self.batch_size, is_origin_size=True)
            
            for j in range(self.batch_size):
                docid = batch_datasets[j]['docid']
                pageid = batch_datasets[j]['pageid']
                tag = batch_datasets[j]['tag']
                output_path = os.path.join(output_dir, 'predictions', '%s_%s_%s.png' % (
                    docid, pageid, tag))
                image_path = batch_datasets[j]['path']
                show_path = image_path.replace('/Tensors/', '/Images/')
                print(show_path)
                image = cv2.imread(show_path)
                texts = processor.content_dict[docid][pageid]['processed_texts']
                
                # 画预测的框
                for box in pred_boxes[j]:
                    pred_texts = []
                    for text in texts:
                        if self.in_table(text, box):
                            pred_texts.append(text)
                    if not pred_texts:
                        continue

                    left = int(min([text['box'][0] for text in pred_texts]))
                    top = int(min([text['box'][1] for text in pred_texts]))
                    right = int(max([text['box'][2] for text in pred_texts]))
                    bottom = int(max([text['box'][3] for text in pred_texts]))
                    cv2.rectangle(image, (left, top), (right, bottom), (238, 192, 126), 2) # blue
                
                cv2.imwrite(output_path, image)
        self.test_sess.close()
        print('Test Finish!')
    
    def deploy_model(self, processor, network, model_path):
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.test_sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        # 读取模型
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        assert(os.path.exists(model_path+'.index'))
        self.saver.restore(self.test_sess, model_path)
        print('read model from %s' % (model_path))
        
        self.network = network
        self.logits = self.network.get_inference(self.images)
        
        pred_outlines = {}
        for i in range(int(processor.n_deploy / self.batch_size)-1):
            # 获取数据并进行数据增强
            batch_images = processor.dataset_producer(
                mode='deploy', index=i*self.batch_size, batch_size=self.batch_size)
            
            [logits] = self.test_sess.run(
                fetches=[self.logits], 
                feed_dict={self.images: batch_images})
            
            # 获得这一批的原始数据
            batch_datasets = processor.deploysets[i*self.batch_size: (i+1)*self.batch_size]

            # 获得预测的框
            pred_boxes = self.get_pred_boxes(logits, batch_datasets, 
                self.batch_size, is_origin_size=True)
            
            for j in range(self.batch_size):
                docid = batch_datasets[j]['docid']
                pageid = batch_datasets[j]['pageid']
                tag = batch_datasets[j]['tag']
                image = batch_datasets[j]['show_image']
                texts = processor.content_dict[docid][pageid]['processed_texts']
                
                # 画预测的框
                pred_outlines[pageid] = []
                for pred_box in pred_boxes[j]:
                    pred_texts = []
                    for text in texts:
                        if text['type'] in ['DIGIT', 'DATE', 'TEXT', 'PAGENO']:
                            if self.in_table(text, pred_box):
                                pred_texts.append(text)
                    if not pred_texts:
                        continue

                    left = int(min([text['box'][0] for text in pred_texts]))
                    top = int(min([text['box'][1] for text in pred_texts]))
                    right = int(max([text['box'][2] for text in pred_texts]))
                    bottom = int(max([text['box'][3] for text in pred_texts]))
                    pred_outlines[pageid].append([left, top, right, bottom])
                
        self.test_sess.close()
        print('Deploy Finish!')

        return pred_outlines
    
    def debug(self, processor, network, backup_dir, output_dir, n_iters=500000, batch_size=128):
        time.sleep(15)
        # 构建会话
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        
        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        
        # 构建模型
        self.network = network
        self.optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
        self.logits = self.network.get_inference(self.images)
       
        self.avg_loss, self.iou_value, self.object_value, self.noobject_value, \
            self.recall_value, self.overlap_value = \
            self.network.get_loss(
                self.images, self.coord_true, self.object_mask,
                self.unpositioned_coord_true, self.unpositioned_object_mask, self.global_step)
    
        # 构建优化器
        with tf.name_scope('optimize'):
            self.optimizer_handle = self.optimizer.minimize(self.avg_loss,
                global_step=self.global_step)
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver.restore(self.sess, '/home/caory/github/PDFInsight/backup/table-v24/model_800.ckpt')
        
        # 模型训练
        process_images = 0
        
        start_time = time.time()
        data_spend, model_spend = 0.0, 0.0
        if not os.path.exists(os.path.join(output_dir, 'debug')):
            os.mkdir(os.path.join(output_dir, 'debug'))
        for n_iter in range(0, n_iters):
            # 获取数据
            st = time.time()
            data = processor.shared_memory.get()
            
            accum_size = 0
            batch_image_indexs = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.image_index_size)], self.image_index_size)
            accum_size += numpy.prod(self.image_index_size)
            batch_images = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.image_all_size)], self.image_all_size)
            accum_size += numpy.prod(self.image_all_size)
            batch_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.coord_true_size)], self.coord_true_size)
            accum_size += numpy.prod(self.coord_true_size)
            batch_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.coord_mask_size)], self.coord_mask_size)
            accum_size += numpy.prod(self.coord_mask_size)
            batch_unpositioned_coord_true = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.up_coord_true_size)], self.up_coord_true_size)
            accum_size += numpy.prod(self.up_coord_true_size)
            batch_unpositioned_object_mask = numpy.reshape(
                data[accum_size: accum_size+numpy.prod(self.up_coord_mask_size)], self.up_coord_mask_size)
            et = time.time()
            data_time = et - st
           
            st = time.time()
            feed_dict={
                self.images: batch_images, 
                self.coord_true: batch_coord_true,
                self.object_mask: batch_object_mask,
                self.unpositioned_coord_true: batch_unpositioned_coord_true,
                self.unpositioned_object_mask: batch_unpositioned_object_mask}
            et = time.time()
            feed_time = et - st
            
            st = time.time()
            [_, avg_loss, iou_value, object_value, noobject_value, recall_value, overlap_value, \
                logits, logits_resort_tf, conf_pred_mask, pred_boxes_tf] = \
                self.sess.run(
                    fetches=[
                        self.optimizer_handle, self.avg_loss, self.iou_value, 
                        self.object_value, self.noobject_value, self.recall_value, self.overlap_value,
                        self.network.logits, self.network.logits_resort, self.network.conf_pred_mask, 
                        self.network.pred_boxes], 
                    feed_dict=feed_dict)
            et = time.time()
            model_time = et - st
            
            print('[%d] data time: %.4f, feed time: %.4f, model time: %.4f' % (
                n_iter, data_time, feed_time, model_time))

            if n_iter >= 0:
                # 获得这批图片的路径
                batch_image_paths = []
                for j in range(self.batch_size):
                    image_path = processor.trainsets[int(round(batch_image_indexs[j]))]['path']
                    batch_image_paths.append(image_path)
                 
                pred_boxes_network = []
                for j in range(self.batch_size):
                    image = cv2.imread(image_path)
                    orig_w, orig_h = image.shape[1], image.shape[0]
                    boxes = []
                    for b in range(self.cell_y_size):
                        for a in range(self.cell_x_size):
                            for n in range(self.n_boxes):
                                if sum(pred_boxes_tf[j,b,a,n]) != 0.0:
                                    [x, y, w, h] = pred_boxes_tf[j,b,a,n,1:5]
                                    x = min(max(0.0, x), 0.9999) * orig_w
                                    y = min(max(0.0, y), 0.9999) * orig_h
                                    w = min(max(0.0, w), 0.9999) * orig_w
                                    h = min(max(0.0, h), 0.9999) * orig_h
                                    left = int(x - w / 2.0)
                                    right = int(x + w / 2.0)
                                    top = int(y - w / 2.0)
                                    bottom = int(y + w / 2.0)
                                    boxes.append([left, right, top, bottom])
                    pred_boxes_network.append(boxes)
                print('network pred_boxes:')
                print(pred_boxes_network)
                
                # 获得预测的框
                pred_boxes = self.get_pred_boxes(
                    logits, batch_image_paths, self.batch_size, is_origin_size=True)
                overlap_numerator = 0
                overlap_denominator = 0
                # 计算每两个预测框之间的overlap面积
                for j in range(self.batch_size):
                    image_path = batch_image_paths[j]
                    print(image_path)
                    for a in range(len(pred_boxes[j])):
                        for b in range(a+1, len(pred_boxes[j])):
                            iou = self.calculate_iou_py(pred_boxes[j][a], pred_boxes[j][b], mode='lrtb')
                            if iou > 0:
                                overlap_numerator += iou
                                print(pred_boxes[j][a], pred_boxes[j][b], iou)
                            overlap_denominator += 1

                overlap = 1.0 * overlap_numerator / overlap_denominator \
                    if overlap_denominator != 0 else 0.0
                print('[%d] overlap: %.6f' % (n_iter, overlap))
                
                if not os.path.exists(os.path.join(output_dir, 'debug', str(n_iter))):
                    os.mkdir(os.path.join(output_dir, 'debug', str(n_iter)))
                for j in range(self.batch_size):
                    image_path = batch_image_paths[j]
                    show_path = image_path.replace('/Tensors/', '/Images/')
                    if os.path.exists(show_path):
                        image = cv2.imread(show_path)
                        for pred_box in pred_boxes[j]:
                            left = int(pred_box[0])
                            right = int(pred_box[1])
                            top = int(pred_box[2])
                            bottom = int(pred_box[3])
                            cv2.rectangle(image, (left, top), (right, bottom), (238, 192, 126), 2) # blue
                        output_path = os.path.join(
                            output_dir, 'debug', str(n_iter), os.path.split(image_path)[1] + '.model.png')
                        cv2.imwrite(output_path, image)
                        for pred_box in pred_boxes_network[j]:
                            left = int(pred_box[0])
                            right = int(pred_box[1])
                            top = int(pred_box[2])
                            bottom = int(pred_box[3])
                            cv2.rectangle(image, (left, top), (right, bottom), (238, 192, 126), 2) # blue
                        output_path = os.path.join(
                            output_dir, 'debug', str(n_iter), os.path.split(image_path)[1] + '.network.png')
                        cv2.imwrite(output_path, image)
                
            process_images += batch_size * self.n_gpus
            
            end_time = time.time()
            spend = end_time - start_time
            
            # 每1轮训练观测一次train_loss    
            print('[%d] train_loss: %.4f, image_nums: %d, spend: %.2f' % (
                n_iter, avg_loss, process_images, spend))
            sys.stdout.flush()
            
            # 每1轮观测一次训练集evaluation
            print('[%d] IOU: %.6f, Object: %.6f, Noobject: %.6f, Recall: %.6f, Overlap: %.6f' % (
                n_iter, iou_value, object_value, noobject_value, recall_value, overlap_value))
            sys.stdout.flush()
            print()
            
        self.sess.close()
    
    def get_pred_boxes(self, logits, batch_datasets, batch_size, is_origin_size=False):
        box_preds = numpy.reshape(logits, (
            batch_size, self.cell_y_size, self.cell_x_size, self.n_boxes, 1+self.n_coord))
        box_preds[:,:,:,:,1:5] = self.get_direct_position_py(box_preds[:,:,:,:,1:5])
        
        pred_boxes = []
        for j in range(batch_size):
            image = batch_datasets[j]['orig_image']
            orig_w, orig_h = image.shape[1], image.shape[0]

            # 获得预测的preds
            preds = []
            for x in range(self.cell_x_size):
                for y in range(self.cell_y_size):
                    for n in range(self.n_boxes):
                        prob = box_preds[j,y,x,n,0]
                        box = box_preds[j,y,x,n,1:5]
                        if prob >= self.network.pred_thresh:
                            preds.append([box, prob])
            
            # 排序并去除多余的box
            preds = sorted(preds, key=lambda x: x[1], reverse=True)
            for x in range(len(preds)):
                if preds[x][1] < self.network.pred_thresh:
                    continue
                for y in range(x+1, len(preds)):
                    iou = self.calculate_iou_py(preds[x][0], preds[y][0])
                    if iou > self.network.nms_thresh:
                        preds[y][1] = 0.0
            
            # 画预测的框
            boxes = []
            for k in range(len(preds)):
                if preds[k][1] >= self.network.pred_thresh:
                    [x, y, w, h] = preds[k][0]
                    if is_origin_size:
                        [x, y, w, h] = self.resize_to_origin(x, y, w, h, orig_w, orig_h)
                    x = min(max(0.0, x), 0.9999) * orig_w
                    y = min(max(0.0, y), 0.9999) * orig_h
                    w = min(max(0.0, w), 0.9999) * orig_w
                    h = min(max(0.0, h), 0.9999) * orig_h
                    left = int(x - w / 2.0)
                    right = int(x + w / 2.0)
                    top = int(y - h / 2.0)
                    bottom = int(y + h / 2.0)
                    boxes.append([left, top, right, bottom])
            pred_boxes.append(boxes)
        
        return pred_boxes
    
    def get_true_boxes(self, batch_datasets, batch_size, mode='origin'):
        true_boxes = []
        for j in range(batch_size):
            image = batch_datasets[j]['orig_image']
            orig_w, orig_h = image.shape[1], image.shape[0]
            label = batch_datasets[j]['label']

            boxes = []
            if mode == 'origin':
                for k in range(self.max_objects):
                    if sum(label[k]) == 0:
                        break
                    [index, left, right, top, bottom] = label[k][0:5]
                    left = int(left * orig_w)
                    right = int(right * orig_w)
                    top = int(top * orig_h)
                    bottom = int(bottom * orig_h)
                    boxes.append([left, top, right, bottom])

            true_boxes.append(boxes)

        return true_boxes

    def get_direct_position_py(self, coord_pred):
        # 计算bx
        offset_x = numpy.reshape(range(0, self.cell_x_size), newshape=(1, 1, self.cell_x_size, 1, 1))
        offset_x = numpy.tile(offset_x, (self.batch_size, self.cell_y_size, 1, self.n_boxes, 1))
        offset_x = numpy.array(offset_x, dtype='float')
        x_pred = (coord_pred[:,:,:,:,0:1] + offset_x) / self.cell_x_size
        
        # 计算by
        offset_y = numpy.reshape(range(0, self.cell_y_size), newshape=(1, self.cell_y_size, 1, 1, 1))
        offset_y = numpy.tile(offset_y, (self.batch_size, 1, self.cell_x_size, self.n_boxes, 1))
        offset_y = numpy.array(offset_y, dtype='float')
        y_pred = (coord_pred[:,:,:,:,1:2] + offset_y) / self.cell_y_size
        
        # 计算pw
        prior_w = numpy.array([10.1, 9.7, 9.5, 9.4, 9.1], dtype='float32')
        prior_w = numpy.reshape(prior_w, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_w = numpy.tile(prior_w, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        w_pred = prior_w * numpy.exp(coord_pred[:,:,:,:,2:3]) / self.cell_x_size
        
        # 计算ph
        prior_h = numpy.array([9.2, 4.2, 2.3, 1.3, 0.5], dtype='float32')
        prior_h = numpy.reshape(prior_h, newshape=(1, 1, 1, self.n_boxes, 1))
        prior_h = numpy.tile(prior_h, (self.batch_size, self.cell_y_size, self.cell_x_size, 1, 1))
        h_pred = prior_h * numpy.exp(coord_pred[:,:,:,:,3:4]) / self.cell_y_size
        
        new_coord_pred = numpy.concatenate([x_pred, y_pred, w_pred, h_pred], axis=4)
        
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
        inter_area = (right - left) * (bottom - top)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        iou = inter_area / (box1_area + box2_area - inter_area + 1e-6) if inter_area >= 0 else 0.0
        
        return iou

    def resize_to_origin(self, x, y, w, h, orig_w, orig_h):
        if orig_w <= orig_h:
            new_x = 1.0 * orig_h / orig_w * (x - (1.0 - 1.0 * orig_w / orig_h) / 2.0)
            new_y = y
            new_w = 1.0 * w * orig_h / orig_w
            new_h = h
        else:
            new_x = x
            new_y = 1.0 * orig_w / orig_h * (y - (1.0 - 1.0 * orig_h / orig_w) / 2.0)
            new_w = w
            new_h = 1.0 * h * orig_w / orig_h

        return [new_x, new_y, new_w, new_h]

    def judge_pred_true_matched(self, pred_box, true_box, texts):
        true_texts, pred_texts = [], []
        for i in range(len(texts)):
            if self.in_table(texts[i], true_box):
                true_texts.append(i)
            if self.in_table(texts[i], pred_box):
                pred_texts.append(i)

        if true_texts == pred_texts:
            return True
        else:
            return False

    def in_table(self, text, box):
        cross_left = max(int(text['box'][0]), int(box[0]))
        cross_top = max(int(text['box'][1]), int(box[1]))
        cross_right = min(int(text['box'][2]), int(box[2]))
        cross_bottom = min(int(text['box'][3]), int(box[3]))
        if cross_left < cross_right and cross_top < cross_bottom:
            return True
        else:
            return False

    def average_gradients(self, tower_grads, tower_evals):
        """
        在CPU中对所有的梯度和评估值进行求平均
        输入1：tower_grads，是一个list，
                [[(grad0_gpu0, var0_gpu0), (grad1_gpu0, var1_gpu0), ...],
                 [(grad0_gpu1, var0_gpu1), (grad1_gpu1, var1_gpu1), ...],
                 ...]
        输入2：tower_evals，是一个list，
                [[eval0_gpu0, eval1_gpu0, ...],
                 [eval0_gpu1, eval1_gpu1, ...],
                 ...]
        输出：avg_grads_and_vars, avg_evals
        """
        # 对梯度进行平均
        avg_grads_and_vars = []
        for grads_and_vars in zip(*tower_grads):
            grads = []
            is_none = False
            for grad, var in grads_and_vars:
                if grad is None:
                    is_none = True
                    break
                exp_grad = tf.expand_dims(grad, 0)
                grads.append(exp_grad)
            if not is_none:
                avg_grads = tf.concat(grads, axis=0)
                avg_grad = tf.reduce_mean(avg_grads, axis=0)
                avg_grads_and_vars.append((avg_grad, var))
            else:
                avg_grads_and_vars.append((None, var))

        # 对评估值进行平均
        avg_evals = []
        for evals in zip(*tower_evals):
            evls = []
            for evl in evals:
                evls.append(tf.expand_dims(evl, 0))
            avg_eval = tf.reduce_mean(tf.concat(evls, axis=0), axis=0)
            avg_evals.append(avg_eval)

        return avg_grads_and_vars, avg_evals

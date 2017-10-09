# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import numpy
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.image import ImageProcessor
from src.model.yolo_v2 import TinyYolo


class TinyYoloTestor:
    
    def test_calculate_loss(self):
        self.batch_size = 1
        self.cell_size = 2
        self.n_boxes = 2
        self.max_objects = 3
        
        coord_pred = numpy.zeros((1, 2, 2, 2, 4))
        coord_pred[0,0,0,0,:] = [0.4, 0.4, 0.1, 0.1]
        coord_pred[0,0,0,1,:] = [0.1, 0.1, 0.1, 0.1]
        coord_pred[0,0,1,0,:] = [0.75, 0.25, 0.1, 0.1]
        coord_pred[0,0,1,1,:] = [0.7, 0.2, 0.1, 0.1]
        coord_pred[0,1,0,0,:] = [0.3, 0.8, 0.1, 0.1]
        coord_pred[0,1,0,1,:] = [0.25, 0.75, 0.1, 0.1]
        coord_pred[0,1,1,0,:] = [0.75, 0.75, 0.1, 0.1]
        coord_pred[0,1,1,1,:] = [0.7, 0.8, 0.1, 0.1]
        
        coord_true = numpy.zeros((1, 2, 2, 3, 4))
        coord_true[0,0,0,0,:] = [0.1, 0.1, 0.1, 0.1]
        coord_true[0,0,0,1,:] = [0.4, 0.4, 0.1, 0.1]
        coord_true[0,0,1,0,:] = [0.75, 0.25, 0.1, 0.1]
        coord_true[0,1,0,0,:] = [0.25, 0.75, 0.1, 0.1]
        coord_true[0,1,1,0,:] = [0.75, 0.75, 0.1, 0.1]
        
        object_mask = numpy.zeros((1, 2, 2, 3))
        object_mask[0,0,0,0] = 1
        object_mask[0,0,0,1] = 1
        object_mask[0,0,1,0] = 1
        object_mask[0,1,0,0] = 1
        object_mask[0,1,1,0] = 1
        
        coord_true_tf = tf.placeholder(
            dtype=tf.float32, shape=[1, 2, 2, 3, 4], name='coord_true_tf')
        coord_pred_tf = tf.placeholder(
            dtype=tf.float32, shape=[1, 2, 2, 2, 4], name='coord_pred_tf')
        object_mask_tf = tf.placeholder(
            dtype=tf.float32, shape=[1, 2, 2, 3], name='object_mask_tf')
        
        coord_pred_iter = tf.tile(
            tf.reshape(coord_pred_tf, shape=[
                self.batch_size, self.cell_size, self.cell_size, self.n_boxes, 1, 4]), 
            [1, 1, 1, 1, self.max_objects, 1])
        
        coord_true_iter = tf.reshape(coord_true_tf, shape=[
                self.batch_size, self.cell_size, self.cell_size, 1, self.max_objects, 4])
        coord_true_iter = tf.tile(coord_true_iter, [1, 1, 1, self.n_boxes, 1, 1])
        
        iou_tensor = self.calculate_iou_tf(coord_pred_iter, coord_true_iter)
        iou_tensor_max = tf.reduce_max(iou_tensor, 3, keep_dims=True) 
        iou_tensor_mask = tf.cast(
            (iou_tensor >= iou_tensor_max), dtype=tf.float32) * tf.reshape(
                object_mask_tf, shape=(
                    self.batch_size, self.cell_size, self.cell_size, 1, self.max_objects, 1))
        
        coord_label = tf.reduce_max(iou_tensor_mask * coord_true_iter, axis=4)
        conf_label = tf.reduce_max(iou_tensor_mask * tf.ones(shape=(
            self.batch_size, self.cell_size, self.cell_size, 
            self.n_boxes, self.max_objects, 1)), axis=4)
            
        sess = tf.Session()
        [output] = sess.run(
            fetches=[conf_label],
            feed_dict={coord_true_tf: coord_true, coord_pred_tf: coord_pred,
                       object_mask_tf: object_mask})
        print(output)
              
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
    
    def test_get_box_pred(self):
            
        label = [[0, 0, 0, 0, 0]] * 5
        label[0] = [0.5, 0.15, 0.8, 0.2, 1]
        label[1] = [0.5, 0.7, 0.1, 0.2, 1]
        label[2] = [0.5, 0.9, 0.6, 0.1, 1]
        
        pred = numpy.zeros(shape=(3,3,6,5)) 
        pred[0,1,4,:] = [-1.6, -1.73, 0.09, -0.09, 1.0]
        # pred[1,0,4,:] = [0.0, 0.0, 0.0, 0.0, 1.0]
        
        image_processor = ImageProcessor(
            'Z:', image_size=96, max_objects_per_image=5, cell_size=3, n_classes=1)
        class_label, class_mask, box_label, object_num = \
           image_processor.process_label(label)
            
        tiny_yolo = TinyYolo(
            n_channel=3, n_classes=1, image_size=96, max_objects_per_image=5,
            box_per_cell=6, object_scala=10, nobject_scala=5,
            coord_scala=10, class_scala=1, batch_size=1)
        
        box_pred = tf.placeholder(
            dtype=tf.float32, shape=[3, 3, 6, 4], name='box_pred')
        box_truth = tf.placeholder(
            dtype=tf.float32, shape=[3, 3, 1, 4], name='box_truth')
        iou_matrix = tiny_yolo.get_box_pred(box_pred)
        sess = tf.Session()
        [output] = sess.run(
            fetches=[iou_matrix],
            feed_dict={box_pred: pred[:,:,:,0:4]})
        sess.close()
        print(output, output.shape)
        
        # 画图
        image = numpy.zeros(shape=(256, 256, 3), dtype='uint8') + 255
        cv2.line(image, (0, int(256/3.0)), (256, int(256/3.0)), (100, 149, 237), 1)
        cv2.line(image, (0, int(256*2.0/3.0)), (256, int(256*2.0/3.0)), (100, 149, 237), 1)
        cv2.line(image, (int(256/3.0), 0), (int(256/3.0), 256), (100, 149, 237), 1)
        cv2.line(image, (int(256*2.0/3.0), 0), (int(256*2.0/3.0), 256), (100, 149, 237), 1)
        for center_x, center_y, w, h, prob in label:
            if prob != 1.0:
                continue
            # 画中心点
            cv2.circle(image, (int(center_x*256), int(center_y*256)), 2, (255, 99, 71), 0)
            # 画真实框
            xmin = int((center_x - w / 2.0) * 256)
            xmax = int((center_x + w / 2.0) * 256)
            ymin = int((center_y - h / 2.0) * 256)
            ymax = int((center_y + h / 2.0) * 256)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 99, 71), 0)
        for x in range(3):
            for y in range(3):
                for n in range(2):
                    [center_x, center_y, w, h, prob] = pred[x, y, n, :]
                    # 画中心点
                    cv2.circle(image, (int(center_x*256), int(center_y*256)), 2, (238, 130, 238), 0)
                    # 画预测框
                    xmin = int((center_x - w / 2.0) * 256)
                    xmax = int((center_x + w / 2.0) * 256)
                    ymin = int((center_y - h / 2.0) * 256)
                    ymax = int((center_y + h / 2.0) * 256)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (238, 130, 238), 0)
            
        plt.imshow(image)
        plt.show()
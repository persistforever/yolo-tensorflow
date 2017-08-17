# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import numpy
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from src.data.image import ImageProcessor
from src.model.yolo_v2 import TinyYolo


class TinyYoloTestor:
    
    def test_get_box_pred(self):
            
        label = [[0, 0, 0, 0, 0]] * 5
        label[0] = [0.5, 0.15, 0.8, 0.2, 1]
        label[1] = [0.5, 0.7, 0.1, 0.2, 1]
        label[2] = [0.5, 0.9, 0.6, 0.1, 1]
        
        pred = numpy.zeros(shape=(3,3,6,5)) 
        pred[1,0,4,:] = [-1.6, -1.73, 0.09, -0.09, 1.0]
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
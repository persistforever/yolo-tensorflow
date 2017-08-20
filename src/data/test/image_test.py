# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import numpy
import random
import matplotlib.pyplot as plt
import cv2
from src.data.image import ImageProcessor


class ImageProcessorTestor:
    
    def test_process_label(self):
        label = [[0, 0, 0, 0, 0]] * 5
        label[0] = [0.5, 0.15, 0.8, 0.2, 1.0]
        label[1] = [0.5, 0.7, 0.6, 0.4, 1.0]
        
        image_processor = ImageProcessor(
            'Z:', image_size=300, max_objects_per_image=5, cell_size=3, n_classes=1)
        class_label, class_mask, box_label, object_mask, nobject_mask, object_num = \
           image_processor.process_label(label)
        print(class_label)
        print(class_mask)
        print(box_label)
        print(object_mask)
        print(nobject_mask)
        print(object_num)
            
        image = numpy.zeros(shape=(300, 300, 3), dtype='uint8') + 255
        cv2.line(image, (0, 100), (300, 100), (100, 149, 237), 1)
        cv2.line(image, (0, 200), (300, 200), (100, 149, 237), 1)
        cv2.line(image, (100, 0), (100, 300), (100, 149, 237), 1)
        cv2.line(image, (200, 0), (200, 300), (100, 149, 237), 1)
        
        for center_x, center_y, w, h, prob in label:
            if prob != 1.0:
                continue
            # 画中心点
            cv2.circle(image, (int(center_x*300), int(center_y*300)), 2, (255, 99, 71), 0)
            # 画框
            xmin = int((center_x - w / 2.0) * 300)
            xmax = int((center_x + w / 2.0) * 300)
            ymin = int((center_y - h / 2.0) * 300)
            ymax = int((center_y + h / 2.0) * 300)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 99, 71), 0)
            
        plt.imshow(image)
        plt.show()
        
    def test_image_resize(self):
        image = numpy.zeros(shape=[525, 828, 3], dtype='uint8') + numpy.array([[[50, 75, 150]]], dtype='uint8')
        plt.imshow(image)
        plt.show()
        
        image_processor = ImageProcessor(
            'Z:', image_size=300, max_objects_per_image=5, cell_size=3, n_classes=1)
        
        new_images = image_processor.image_resize(numpy.array([image]))
        
        plt.imshow(new_images[0])
        plt.show()
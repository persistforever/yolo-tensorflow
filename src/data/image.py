# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import pickle
import numpy
import random
import matplotlib.pyplot as plt
import platform
import cv2
from threading import Thread, Lock, Semaphore
if 'Windows' in platform.platform():
    from queue import Queue
elif 'Linux' in platform.platform():
    import Queue


class ImageProcessor:
    
    def __init__(self, directory, image_size=288, max_objects_per_image=20):
        # 参数赋值
        self.image_size = image_size
        self.max_objects = max_objects_per_image
        self._load_images(directory)
        self._shuffle_images()
        
    def _load_dataset(self, filename, n_thread=10):
        # 读取训练集/验证集/测试集
        info_list = Queue(maxsize=200000)
        dataset = Queue(maxsize=200000)
        
        # 读取info_list
        with open(filename, 'r') as fo:
            for line in fo:
                infos = line.strip().split(' ')
                info_list.put(infos)
        
        def _process(name):
            
            while not info_list.empty():
                
                infos = info_list.get()
                image_path = infos[0]
                label_infos = infos[1:]
                
                # 读取图像
                image = cv2.imread(image_path)
                [image_h, image_w, _] = image.shape
                image = cv2.resize(image, (self.image_size, self.image_size))
                
                # 处理 label
                i, n_objects = 0, 0
                label = [[0, 0, 0, 0, 0]] * self.max_objects
                while i < len(label_infos) and n_objects < self.max_objects:
                    xmin = int(label_infos[i])
                    ymin = int(label_infos[i+1])
                    xmax = int(label_infos[i+2])
                    ymax = int(label_infos[i+3])
                    class_index = int(label_infos[i+4])
                    
                    # 转化成 center_x, center_y, w, h
                    center_x = (1.0 * (xmin + xmax) / 2.0) / image_w
                    center_y = (1.0 * (ymin + ymax) / 2.0) / image_h
                    w = (1.0 * (xmax - xmin)) / image_w
                    h = (1.0 * (ymax - ymin)) / image_h
                    
                    label[n_objects] = [center_x, center_y, w, h, class_index]
                    i += 5
                    n_objects += 1
                    
                dataset.put([image, label])
                
        # 以多线程的方式进行数据预处理
        thread_list = []
        for i in range(n_thread):
            thread = Thread(target=_process, args=(i,))
            thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
        
        # 处理dataset，将其分解成images和labels
        images, labels = [], []
        while not dataset.empty():
            image, label = dataset.get()
            images.append(image)
            labels.append(label)
        
        images = numpy.array(images, dtype='float32')
        labels = numpy.array(labels, dtype='int32')
        
        return images, labels
    
    def _load_images(self, directory):
            
        # 读取训练集
        train_file = os.path.join(directory, 'train.txt')
        self.train_images, self.train_labels = self._load_dataset(train_file, n_thread=5)
        self.n_train = self.train_images.shape[0]
        
        # 读取验证集
        valid_file = os.path.join(directory, 'valid.txt')
        self.valid_images, self.valid_labels = self._load_dataset(valid_file, n_thread=5)
        self.n_valid = self.valid_images.shape[0]
        
        # 读取测试集
        test_file = os.path.join(directory, 'test.txt')
        self.test_images, self.test_labels = self._load_dataset(test_file, n_thread=5)
        self.n_test = self.valid_images.shape[0]
        
        print('train images size: (%d,%d,%d,%d), train labels size: (%d,%d,%d)' % (
            self.train_images.shape[0], self.train_images.shape[1], self.train_images.shape[2],
            self.train_images.shape[3], self.train_labels.shape[0], self.train_labels.shape[1],
            self.train_labels.shape[2]))
        print('valid images size: (%d,%d,%d,%d), valid labels size: (%d,%d,%d)' % (
            self.valid_images.shape[0], self.valid_images.shape[1], self.valid_images.shape[2],
            self.valid_images.shape[3], self.valid_labels.shape[0], self.valid_labels.shape[1],
            self.valid_labels.shape[2]))
        print('test images size: (%d,%d,%d,%d), test labels size: (%d,%d,%d)' % (
            self.test_images.shape[0], self.test_images.shape[1], self.test_images.shape[2],
            self.test_images.shape[3], self.test_labels.shape[0], self.test_labels.shape[1],
            self.test_labels.shape[2]))
        print()
        sys.stdout.flush()
        
    def _shuffle_images(self):
        # 打乱训练集
        index = list(range(self.train_images.shape[0]))
        random.shuffle(index)
        self.train_images = self.train_images[index]
        self.train_labels = self.train_labels[index]
        
        # 打乱验证集
        index = list(range(self.valid_images.shape[0]))
        random.shuffle(index)
        self.valid_images = self.valid_images[index]
        self.valid_labels = self.valid_labels[index]
        
        # 打乱测试集
        index = list(range(self.test_images.shape[0]))
        random.shuffle(index)
        self.test_images = self.test_images[index]
        self.test_labels = self.test_labels[index]
        
    def data_augmentation(self, images, mode='train', flip=False, 
                          crop=False, crop_shape=(24,24,3), whiten=False, 
                          noise=False, noise_mean=0, noise_std=0.01):
        # 图像切割
        if crop:
            if mode == 'train':
                images = self._image_crop(images, shape=crop_shape)
            elif mode == 'test':
                images = self._image_crop_test(images, shape=crop_shape)
        # 图像翻转
        if flip:
            images = self._image_flip(images)
        # 图像白化
        if whiten:
            images = self._image_whitening(images)
        # 图像噪声
        if noise:
            images = self._image_noise(images, mean=noise_mean, std=noise_std)
            
        return images
    
    def _image_crop(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = numpy.random.randint(old_image.shape[0] - shape[0] + 1)
            top = numpy.random.randint(old_image.shape[1] - shape[1] + 1)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_crop_test(self, images, shape):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            left = int((old_image.shape[0] - shape[0]) / 2)
            top = int((old_image.shape[1] - shape[1]) / 2)
            new_image = old_image[left: left+shape[0], top: top+shape[1], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def _image_flip(self, images):
        # 图像翻转
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_whitening(self, images):
        # 图像白化
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
            images[i,:,:,:] = new_image
        
        return images
    
    def _image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            new_image = old_image
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i,:,:,:] = new_image
        
        return images
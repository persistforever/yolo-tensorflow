# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import time
import pickle
import math
import numpy
import random
import platform
import cv2
import multiprocessing as mp


class ImageProcessor:
    
    def __init__(self, directory, image_size, max_objects_per_image, cell_size,
                 n_classes, batch_size):
        # 参数赋值
        self.image_size = image_size
        self.max_objects = max_objects_per_image
        self.cell_size = cell_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        
        self.load_images_labels(directory)
    
    def load_images_labels(self, directory):
        if os.path.exists(directory):
            # 读取训练集
            train_file = os.path.join(directory, 'train.txt')
            self.trainsets = self.load_datasets_whole(train_file)
            self.n_train = len(self.trainsets)
            
            # 读取验证集
            valid_file = os.path.join(directory, 'valid.txt')
            self.validsets = self.load_datasets_whole(valid_file)
            self.n_valid = len(self.validsets)
            
            # 读取测试集
            test_file = os.path.join(directory, 'test.txt')
            self.testsets = self.load_datasets_whole(test_file)
            self.n_test = len(self.testsets)
            
            print('number of train sets: %d' % (self.n_train))
            print('number of valid sets: %d' % (self.n_valid))
            print('number of test sets: %d' % (self.n_test))
            sys.stdout.flush()
        
    def load_datasets_whole(self, filename):
        # 读取训练集/验证集/测试集
        image_paths = []
        datasets = []
        
        # 读取image_paths
        with open(filename, 'r') as fo:
            for line in fo:
                image_path = line.strip()
                image_paths.append(image_path)
                
        for image_path in image_paths:    
            label_path = image_path.replace('Images', 'Labels')
            label_path = label_path.replace('.png', '.txt')
            label_path = label_path.replace('.jpg', '.txt')
            
            label = [[0, 0, 0, 0, 0]] * self.max_objects
            n_objects = 0
                    
            with open(label_path, 'r') as fo:
                for line in fo:
                    infos = line.strip().split(' ')
                    
                    index = float(infos[0])
                    x = float(infos[1])
                    y = float(infos[2])
                    w = float(infos[3])
                    h = float(infos[4])
                    
                    left = x - w / 2.0
                    right = x + w / 2.0
                    top = y - h / 2.0
                    bottom = y + h / 2.0
                    
                    label[n_objects] = [left, right, top, bottom, index]
                    n_objects += 1
                        
            datasets.append([image_path, label])
        
        return datasets

    def dataset_producer(self, dataset):
        while True:
            # 获取数据并进行数据增强
            batch_image_paths, batch_labels = self.get_random_batch(
                self.trainsets, self.batch_size)
            batch_images, batch_labels = self.data_augmentation(
                batch_image_paths, batch_labels, mode='train',
                flip=True, whiten=False, resize=True, jitter=0.2)
            batch_coord_true, batch_class_true, batch_object_mask = \
                self.process_batch_labels(batch_labels)
            dataset.put([batch_images, batch_coord_true, batch_class_true, batch_object_mask])
    
    def get_random_batch(self, dataset, batch_size):
        batch_image_paths, batch_labels = [], []
            
        for i in range(batch_size):
            index = random.randint(0, len(dataset)-1)
            image_path, label = dataset[index]
            batch_image_paths.append(image_path)
            batch_labels.append(label)
            
        return batch_image_paths, batch_labels
            
    def get_index_batch(self, dataset, index, batch_size):
        batch_image_paths, batch_labels = [], []
            
        for j in range(batch_size):
            image_path, label = dataset[index+j]
            batch_image_paths.append(image_path)
            batch_labels.append(label)
            
        return batch_image_paths, batch_labels
    
    def data_augmentation(self, image_paths, labels, mode='train', 
                          resize=False, jitter=0.2, flip=False, whiten=False):
        new_images, new_labels = [], []

        for image_path, label in zip(image_paths, labels):
            image = cv2.imread(image_path)
            # 图像尺寸变换
            if resize:
                image, label = self.image_resize(
                    image, label, jitter=jitter, mode=mode)
            # 图像翻转
            if flip:
                image, label = self.image_flip(image, label, mode=mode)
            # 图像白化
            if whiten:
                image = self.image_whitening(image)
                
            new_images.append(image)
            new_labels.append(label)
        
        new_images = numpy.array(new_images, dtype='uint8')
        new_labels = numpy.array(new_labels, dtype='float32')
         
        return new_images, new_labels
    
    def image_flip(self, image, label, mode='train'):
        # 图像翻转
        if mode == 'train':
            old_image = image
            if numpy.random.random() < 0.5:
                new_image = cv2.flip(old_image, 1)
            else:
                new_image = old_image
            
            # 重新计算box label
            for j in range(len(label)):
                if sum(label[j]) == 0:
                    break
                right = 1.0 - label[j][0]
                left = 1.0 - label[j][1]
                label[j][0] = left
                label[j][1] = right
        else:
            new_image = image
        
        return new_image, label
    
    def image_whitening(self, image):
        # 图像白化
        old_image = image
        new_image = (old_image - numpy.mean(old_image)) / numpy.std(old_image)
        
        return new_image
    
    def image_resize(self, image, label, jitter=0.2, mode='train'):
        resized_w, resized_h = int(self.image_size), int(self.image_size)
        orig_w, orig_h = image.shape[1], image.shape[0]
        
        if mode == 'train':
            dw, dh = image.shape[1] * jitter, image.shape[0] * jitter
            
            # 随机宽高比
            new_ar = 1.0 * (orig_w + random.randint(-int(dw), int(dw))) / \
                (orig_h + random.randint(-int(dh), int(dh)))
            # 随机缩放尺度
            scala = random.random() * (1.2 - 0.8) + 0.8
            
            # 新图像事原图像的scala的缩放，并使新图像的比例为new_ar
            if new_ar < 1.0:
                nh = scala * resized_h
                nw = nh * new_ar
            else:
                nw = scala * resized_w
                nh = nw / new_ar
            nw, nh = int(nw), int(nh)
            
            temp_image = cv2.resize(image, dsize=(nw, nh))
            
            if resized_w > nw:
                dx = random.randint(0, resized_w - nw)
                old_sx, old_ex = 0, nw
                new_sx, new_ex = dx, dx + nw
            else:
                dx = random.randint(0, nw - resized_w)
                old_sx, old_ex = dx, dx + resized_w
                new_sx, new_ex = 0, resized_w
                
            if resized_h > nh:
                dy = random.randint(0, resized_h - nh)
                old_sy, old_ey = 0, nh
                new_sy, new_ey = dy, dy + nh
            else:
                dy = random.randint(0, nh - resized_h)
                old_sy, old_ey = dy, dy + resized_h
                new_sy, new_ey = 0, resized_h
            
            new_image = numpy.zeros(shape=(resized_h, resized_w, 3)) + 128
            new_image[new_sy: new_ey, new_sx: new_ex, :] = \
                temp_image[old_sy: old_ey, old_sx: old_ex, :]
        
            # 重新计算labels
            new_label = [[0, 0, 0, 0, 0]] * self.max_objects
            n = 0
            
            for j in range(len(label)):
                if sum(label[j]) == 0:
                    break
                        
                [left, right, top, bottom, index] = label[j]
                    
                if resized_w > nw:
                    new_left = (1.0 * left * nw + dx) / resized_w
                    new_right = (1.0 * right * nw + dx) / resized_w
                else:
                    new_left = (1.0 * left * nw - dx) / resized_w
                    new_right = (1.0 * right * nw - dx) / resized_w
                    
                if resized_h > nh:
                    new_top = (1.0 * top * nh + dy) / resized_h
                    new_bottom = (1.0 * bottom * nh + dy) / resized_h
                else:
                    new_top = (1.0 * top * nh - dy) / resized_h
                    new_bottom = (1.0 * bottom * nh - dy) / resized_h
                    
                new_left = min(max(0.0, new_left), 1.0 - 1e-6)
                new_right = max(0.0, min(new_right, 1.0 - 1e-6))
                new_top = min(max(0.0, new_top), 1.0 - 1e-6)
                new_bottom = max(0.0, min(new_bottom, 1.0 - 1e-6))
                
                if new_right > new_left and new_bottom > new_top:
                    new_label[n] = [new_left, new_right, new_top, new_bottom, index]
                    n += 1
        else:
            new_w, new_h = orig_w, orig_h
            if resized_w / orig_w  < resized_h / orig_h:
                new_w = int(resized_w)
                new_h = int(resized_h * new_w / resized_w)
            else:
                new_h = int(resized_h)
                new_w = int(resized_w * new_h / resized_h)
            
            temp_image = cv2.resize(image, dsize=(new_w, new_h))
            
            dx = int((resized_w - new_w) / 2.0)
            dy = int((resized_h - new_h) / 2.0)
            
            sx, ex = dx, dx + new_w
            sy, ey = dy, dy + new_h
            
            new_image = numpy.zeros(shape=(resized_h, resized_w, 3)) + 128
            new_image[sy:ey, sx:ex] = temp_image
            
            new_label = label
        
        return new_image, new_label
    
    def image_crop(self, images, padding=20):
        # 图像切割
        new_images = []
        for i in range(images.shape[0]):
            old_image = images[i,:,:,:]
            old_image = numpy.lib.pad(
                old_image, ((padding, padding), (padding, padding), (0,0)),
                'constant', constant_values=((0, 0), (0, 0), (0, 0)))
            left = numpy.random.randint(int(padding*2))
            top = numpy.random.randint(int(padding*2))
            new_image = old_image[left: left+images.shape[1], top: top+images.shape[2], :]
            new_images.append(new_image)
        
        return numpy.array(new_images)
    
    def image_noise(self, images, mean=0, std=0.01):
        # 图像噪声
        for i in range(len(images)):
            old_image = images[i]
            new_image = old_image
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    for k in range(image.shape[2]):
                        new_image[i, j, k] += random.gauss(mean, std)
            images[i] = new_image
        
        return images
    
    def process_label(self, label):
        
        # true_label and mask in 包围框标记
        coord_true = numpy.zeros(
            shape=(self.cell_size, self.cell_size, self.max_objects, 4),
            dtype='float32')
        class_true = numpy.zeros(
            shape=(self.cell_size, self.cell_size, self.max_objects, self.n_classes),
            dtype='float32')
        object_mask = numpy.zeros(
            shape=(self.cell_size, self.cell_size, self.max_objects),
            dtype='float32')
        object_nums = numpy.zeros(
            shape=(self.cell_size, self.cell_size),
            dtype='int')
        
        for j in range(self.max_objects):
            
            [left, right, top, bottom, index] = label[j]
            
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            w = right - left
            h = bottom - top
            index = int(index)
            
            if left != 0.0 and right != 0.0 and top != 0.0 and bottom != 0.0:
                # 计算包围框标记
                center_cell_x = int(math.floor(self.cell_size * center_x))
                center_cell_y = int(math.floor(self.cell_size * center_y))
                coord_true[center_cell_y, center_cell_x, 
                           object_nums[center_cell_y, center_cell_x]] = \
                           numpy.array([center_x, center_y, w, h])
                object_mask[center_cell_y, center_cell_x, 
                            object_nums[center_cell_y, center_cell_x]] = 1.0
                object_nums[center_cell_y, center_cell_x] += 1
                class_vector = numpy.zeros((self.n_classes, ))
                class_vector[index] = 1.0
                class_true[center_cell_y, center_cell_x, 
                           object_nums[center_cell_y, center_cell_x]] = \
                           class_vector
                            
        return coord_true, class_true, object_mask
    
    def process_batch_labels(self, batch_labels):
        batch_coord_true, batch_class_true, batch_object_mask = [], [], []
            
        for i in range(len(batch_labels)):
            coord_true, class_true, object_mask = self.process_label(batch_labels[i])
            batch_coord_true.append(coord_true)
            batch_class_true.append(class_true)
            batch_object_mask.append(object_mask)
        
        batch_coord_true = numpy.array(batch_coord_true, dtype='float32')
        batch_class_true = numpy.array(batch_class_true, dtype='float32')
        batch_object_mask = numpy.array(batch_object_mask, dtype='float32')
        
        return batch_coord_true, batch_class_true, batch_object_mask
        
    def _shuffle_datasets(self, images, labels):
        index = list(range(images.shape[0]))
        random.shuffle(index)
        
        return images[index], labels[index]

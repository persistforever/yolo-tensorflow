# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/03/10
# description: data processing module in object detection
from __future__ import print_function
import sys
import os
import time
import json
import math
import numpy
import random
import platform
import cv2
import multiprocessing as mp
from multiprocessing.sharedctypes import Array, Value
from ctypes import c_double, cast, POINTER

"""
数据处理类：对数据进行预处理、数据扩增等过程
"""
class Processor:
    
    def __init__(self, 
        image_x_size, 
        image_y_size, 
        max_objects,
        n_classes, 
        cell_x_size, 
        cell_y_size,
        batch_size, 
        n_channel, 
        n_processes, 
        n_iters, 
        buffer_size):

        # 参数赋值
        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.max_objects = max_objects
        self.n_classes = n_classes + 1
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.n_processes = n_processes
        self.n_iters = n_iters
        self.buffer_size = buffer_size
        
        self.index_size = (self.batch_size)
        self.image_size = (self.batch_size, self.image_y_size, self.image_x_size, 3)
        self.coord_true_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects, 4)
        self.object_mask_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects)
        self.class_true_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects, n_classes)
        self.unpos_coord_true_size = (self.batch_size, self.max_objects, 4)
        self.unpos_object_mask_size = (self.batch_size, self.max_objects)
        self.object_nums_size = (self.batch_size, self.cell_y_size, self.cell_x_size)
        
        self.dataset_size = sum([numpy.prod(t) for t in [
            self.index_size, self.image_size, self.coord_true_size,
            self.object_mask_size, self.class_true_size, 
            self.unpos_coord_true_size, self.unpos_object_mask_size, 
            self.object_nums_size]])
        
        
    def init_datasets(self, mode, train_image_paths_file=None,
        test_image_paths_file=None): 
        """
        初始化数据集
        输入1：mode - 训练/验证/测试/应用
        输入2：train_image_paths_file - 训练数据图片存储的路径文件
        输入3：test_image_paths_file - 测试数据图片存储的路径文件
        """
        # 根据mode进行image_processor的初始化，并判断参数是否出错
        if mode == 'train':
            if train_image_paths_file is None and test_image_paths_file is None:
                raise('ERROR: wrong parameters in initialization!')
        elif mode == 'test':
            if test_image_paths_file is None:
                raise('ERROR: wrong parameters in initialization!')
        else:
            raise('ERROR: wrong mode in initialization!')

        if mode == 'train':
            self.load_datasets('train', image_paths_file=train_image_paths_file)
            self.load_datasets('valid', image_paths_file=test_image_paths_file)
            self.shared_memory = SharedMemory(self.buffer_size, self.dataset_size)
            print('finish apply shared memory ...')
            sys.stdout.flush()
        elif mode == 'test':
            self.load_datasets('test', image_paths_file=test_image_paths_file)
    
    def load_datasets(self, mode, image_paths_file=None, items=None):
        """
        读取数据集
        输入1：mode - 训练/验证/测试/应用
        输入2：image_paths_files - 数据图片存储的路径文件
        输入3：items - 数据的list，每一个元素是一个数据item
        """
        # 在训练/验证/测试/应用不同模式下读取datasets
        datasets = []
        
        # 判断参数是否出错
        if mode in ['train', 'valid', 'test']:
            if image_paths_file is None:
                raise('ERROR: wrong parameters in load_datasets!')
        else:
            raise('ERROR: wrong mode in load_datasets!')

        if mode == 'train':
            datasets = self.init_subdataset(image_paths_file)
            self.trainsets = datasets
            self.n_train = len(self.trainsets)
            print('number of train images: %d' % (self.n_train))
        elif mode == 'valid':
            datasets = self.init_subdataset(image_paths_file)
            self.validsets = datasets
            self.n_valid = len(self.validsets)
            print('number of valid images: %d' % (self.n_valid))
        elif mode == 'test':
            datasets = self.init_subdataset(image_paths_file)
            self.testsets = datasets
            self.n_test = len(self.testsets)
            print('number of test images: %d' % (self.n_test))

    def init_subdataset(self, image_paths_file):
        datasets = []
        image_infos = self._load_image_paths_from_file(image_paths_file)
                
        # 组织datasets, datasets是一个item的list
        for i in range(len(image_infos)):
            [image_path, image_name] = image_infos[i]
            label_path = image_path.replace('Images', 'Labels')
            label_path = label_path.replace('.jpg', '.txt')
           
            item = {'image_name': image_name, 'image_path': image_path, 'label_path': label_path}
            item['label'] = self._get_label_from_path(item['label_path'])
            item['image'] = self._get_image_from_path(item['image_path'])
            
            datasets.append(item)
        
        return datasets
               
    def dataset_producer_based_shm(self, produce_index=0):
        """
        基于共享内存的方法生产数据
        """
        i_basic, i_adapt = 0, 0
        while True:
            batch_indexs, batch_images, batch_labels = \
                self.get_random_batch(self.trainsets, self.batch_size)
            
            batch_indexs = numpy.array(batch_indexs, dtype='float32')
            batch_images = self.convert_batch_images(batch_images)
            batch_images = numpy.array(batch_images, dtype='float32')
            
            # 数据增强
            batch_coord_true, batch_object_mask, batch_class_true, \
                batch_unpos_coord_true, batch_unpos_object_mask, batch_object_nums = \
                self.convert_batch_labels(batch_labels)
            
            # 改变尺寸存入shared_memory
            batch_indexs = batch_indexs.flatten()
            batch_images = batch_images.flatten()
            batch_coord_true = batch_coord_true.flatten()
            batch_object_mask = batch_object_mask.flatten()
            batch_class_true = batch_class_true.flatten()
            batch_unpos_coord_true =  batch_unpos_coord_true.flatten()
            batch_unpos_object_mask = batch_unpos_object_mask.flatten()
            batch_object_nums = batch_object_nums.flatten()
            
            dataset = numpy.concatenate([
                batch_indexs, batch_images, batch_coord_true, batch_object_mask, batch_class_true,
                batch_unpos_coord_true, batch_unpos_object_mask, batch_object_nums], axis=0)
            self.shared_memory.put(dataset)
            
    def get_random_batch(self, datasets, batch_size):
        """
        获取随机一个batch的数据
        输入1：datasets - 整个数据集
        输入2：batch_size - 该batch的大小
        输出1：batch_indexs - 每条数据在整个数据集中的index
        输出2：batch_images - 每条数据的图片
        输出3：batch_labels - 每条数据的标签
        """
        batch_indexs, batch_images, batch_labels = [], [], []
       
        for i in range(batch_size):
            valid_indexs = range(self.n_train)
            index = random.choice(valid_indexs)
            item = datasets[index]
            image = item['image']
            label = item['label']
            batch_indexs.append(index)
            batch_images.append(image)
            batch_labels.append(label)
        
        return batch_indexs, batch_images, batch_labels

    def dataset_producer(self, mode, indexs):
        # 直接从内存获取一个batch的数据
        if mode == 'train':
            dictionary = self.trainsets
        elif mode == 'valid':
            dictionary = self.validsets
        elif mode == 'test':
            dictionary = self.deploysets

        batch_images, batch_datasets = [], []
        for index in indexs:
            batch_images.append(dictionary[index]['image'])
            batch_datasets.append(dictionary[index])
        
        batch_images = self.convert_batch_images(batch_images)
        batch_images = numpy.array(batch_images, dtype='float32')
        return batch_images, batch_datasets

    def convert_batch_images(self, batch_images):
        """
        将一个batch的images从list转化成numpy.array
        """
        new_batch_images = []

        for image in batch_images:
            orig_h, orig_w = image.shape[0], image.shape[1]
            canvas_image = numpy.zeros((self.image_y_size, self.image_x_size, 3), dtype='int32') + 255
            if 1.0 * orig_h / orig_w >= 1.0 * self.image_y_size / self.image_x_size:
                new_h = self.image_y_size
                new_w = int(round(1.0 * new_h / orig_h * orig_w))
                image = cv2.resize(image, (new_w, new_h))
                start_x = int(round((self.image_x_size - new_w) / 2.0))
                canvas_image[:, start_x: start_x+new_w, :] = image
            else:
                new_w = self.image_x_size
                new_h = int(round(1.0 * new_w / orig_w * orig_h))
                image = cv2.resize(image, (new_w, new_h))
                start_y = int(round((self.image_y_size - new_h) / 2.0))
                canvas_image[start_y: start_y+new_h, :, :] = image
            print(canvas_image.shape)
            new_batch_images.append(canvas_image)

        return new_batch_images

    def convert_batch_labels(self, batch_labels):
        """
        将一个batch的label从list转化成numpy.array
        """
        batch_coord_true, batch_object_mask, batch_class_true, batch_unpos_coord_true, \
            batch_unpos_object_mask, batch_object_nums = [], [], [], [], [], []
            
        for i in range(len(batch_labels)):
            coord_true, object_mask, class_true, unpos_coord_true, unpos_object_mask, object_nums = \
                self._process_label(batch_labels[i])
            batch_coord_true.append(coord_true)
            batch_object_mask.append(object_mask)
            batch_class_true.append(class_true)
            batch_unpos_coord_true.append(unpos_coord_true)
            batch_unpos_object_mask.append(unpos_object_mask)
            batch_object_nums.append(object_nums)
        
        batch_coord_true = numpy.array(batch_coord_true, dtype='float32')
        batch_object_mask = numpy.array(batch_object_mask, dtype='float32')
        batch_class_true = numpy.array(batch_class_true, dtype='float32')
        batch_unpos_coord_true = numpy.array(batch_unpos_coord_true, dtype='float32')
        batch_unpos_object_mask = numpy.array(batch_unpos_object_mask, dtype='float32')
        batch_object_nums = numpy.array(batch_object_nums, dtype='float32')
        
        return batch_coord_true, batch_object_mask, batch_class_true, \
            batch_unpos_coord_true, batch_unpos_object_mask, batch_object_nums

    def _process_label(self, label):
        """
        处理所有network部分所需要的label
        """
        coord_true = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size, self.max_objects, 8),
            dtype='float32')
        object_mask = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size, self.max_objects),
            dtype='float32')
        class_true = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size, self.max_objects, self.n_classes),
            dtype='float32')
        unpos_coord_true = numpy.zeros(
            shape=(self.max_objects, 8),
            dtype='float32')
        unpos_object_mask = numpy.zeros(
            shape=(self.max_objects, ),
            dtype='float32')
        object_nums = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size),
            dtype='int32')
        
        for j in range(self.max_objects):
            
            [index, in_x, in_y, in_w, in_h] = label[j]
            
            if not (in_x == 0.0 and in_y == 0.0 and in_w == 0.0 and in_h == 0.0):
                # 计算包围框标记
                center_cell_x = min(int(self.cell_x_size * in_x), self.cell_x_size-1)
                center_cell_y = min(int(self.cell_y_size * in_y), self.cell_y_size-1)
                
                offl = (in_x - in_w / 2.0) - (out_x - out_w / 2.0)
                offt = (in_y - in_h / 2.0) - (out_y - out_h / 2.0)
                offr = (out_x + out_w / 2.0) - (in_x + in_w / 2.0)
                offb = (out_y + out_h / 2.0) - (out_y + out_h / 2.0)
                coord_true[center_cell_y, center_cell_x, 
                    object_nums[center_cell_y, center_cell_x],:] = numpy.array(
                    [in_x, in_y, in_w, in_h])
                object_mask[center_cell_y, center_cell_x,
                    object_nums[center_cell_y, center_cell_x]] = 1.0
                
                object_nums[center_cell_y, center_cell_x] += 1

                unpos_coord_true[j,:] = numpy.array([in_x, in_y, in_w, in_h])
                unpos_object_mask[j] = 1.0
                
                class_true[center_cell_y, center_cell_x, 
                    object_nums[center_cell_y, center_cell_x], index] = 1.0
                
        return coord_true, object_mask, class_true, \
            unpos_coord_true, unpos_object_mask, object_nums

    def _load_image_paths_from_file(self, image_paths_file):
        """
        从文件中读取所有图片的路径
        输入：image_paths_file - 图片路径文件
        输出：image_paths - 图片路径list，每一个元素包含图片路径和图片名
        """
        image_paths = []
        
        with open(image_paths_file, 'r') as fo:
            for line in fo:
                image_path = line.strip()
                if not os.path.exists(image_path):
                    print(image_path)
                    raise('ERROR: image path not exists!')
                file_name = os.path.split(image_path)[1]
                image_name = os.path.splitext(file_name)[0]
                image_paths.append([image_path, image_name])

        return image_paths

    def _get_label_from_path(self, label_path):
        """
        根据标签路径读取标签
        输入：label_path_file - 标签路径
        输出：labels - 图片标签list
        """
        new_label = numpy.zeros((self.max_objects, 5), dtype='float32')
        n_object = 0
            
        with open(label_path, 'r') as fo:
            for line in fo:
                infos = line.strip().split(' ')
                
                index = float(infos[0])
                in_x = float(infos[1])
                in_y = float(infos[2])
                in_w = float(infos[3])
                in_h = float(infos[4])
            
                new_label[n_object,:] = numpy.array([index, in_x, in_y, in_w, in_h], dtype='float32')
                n_object += 1

                if n_object >= self.max_objects:
                    break
        
        return new_label

    def _get_image_from_path(self, image_path):
        image = cv2.imread(image_path)
        image = numpy.array(image / 255.0, dtype='float32')
        return image


"""
共享内存类：生产者消费者模式下，用于存储和传递数据的共享内存区域
"""
class SharedMemory:
    
    def __init__(self, buffer_size, dataset_size):
        self.buffer_size = buffer_size
        self.dataset_size = dataset_size
        self.put_index = Value('i', 0)
        self.get_index = Value('i', 0)
        self.put_lock = mp.Lock()

        self.cdatasets = Array('d', [0.0] * self.buffer_size * self.dataset_size)
        self.cbuffer = self.cdatasets._obj._wrapper
        
    def put(self, dataset):
        """
        将数据写入共享内存
        输入：dataset - 一个batch的数据（已经flatten）
        """
        # 向共享内存中生产数据
        with self.put_lock:
            while self.put_index.value - self.get_index.value >= self.buffer_size - 1:
                time.sleep(0.1)
            index = self.put_index.value % self.buffer_size
            buffer_ptr = cast(self.cbuffer.get_address() + index * self.dataset_size * 8, POINTER(c_double))
            data = numpy.ctypeslib.as_array(buffer_ptr, shape=(self.dataset_size, ))
            data[:] = dataset
            self.put_index.value += 1

    def get(self):
        """
        从共享内存读取数据
        输出：data - 一个batch的数据（已经flatten）
        """
        # 向共享内存中消费数据
        while self.put_index.value - self.get_index.value <= 0:
            time.sleep(0.1)
        index = self.get_index.value % self.buffer_size
        buffer_ptr = cast(self.cbuffer.get_address() + index * self.dataset_size * 8, POINTER(c_double))
        data = numpy.ctypeslib.as_array(buffer_ptr, shape=(self.dataset_size, ))
        self.get_index.value += 1

        return data


"""
共享区域类：生产者消费者模式下，用于存储和传递数据的共享内存区域
"""
class SharedBlock:
    
    def __init__(self, dataset_size):
        self.dataset_size = dataset_size
        self.index = Value('i', 0)
        self.get_lock = mp.Lock()
        self.index_lock = mp.Lock()

        self.cdatasets = Array('d', [0.0] * 2 * self.dataset_size)
        self.cbuffer = self.cdatasets._obj._wrapper

        init_array = numpy.ones((dataset_size, ), dtype='float32')
        self.put(init_array)
        self.put(init_array)
        
    def put(self, dataset):
        """
        将数据写入共享内存
        输入：dataset - 写入新的数据
        """
        # 向共享内存中生产数据
        index = self.index.value % 2
        buffer_ptr = cast(self.cbuffer.get_address() + index * self.dataset_size * 8, POINTER(c_double))
        data = numpy.ctypeslib.as_array(buffer_ptr, shape=(self.dataset_size, ))
        data[:] = dataset
        with self.index_lock:
            self.index.value += 1

    def get(self):
        """
        从共享内存读取数据
        输出：data - 读取旧的数据
        """
        # 向共享内存中消费数据
        with self.get_lock:
            with self.index_lock:
                index = (self.index.value + 1) % 2
            buffer_ptr = cast(self.cbuffer.get_address() + index * self.dataset_size * 8, POINTER(c_double))
            data = numpy.ctypeslib.as_array(buffer_ptr, shape=(self.dataset_size, ))

        return data

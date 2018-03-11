# -*- coding: utf8 -*-
# author: ronniecao
# time: 2017/12/19
# description: data processing module in table detection
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
    
    def __init__(self, element_vector_path,
        image_x_size, image_y_size, max_objects_per_image, cell_x_size, cell_y_size,
        n_classes, batch_size, n_channel, n_iters, n_processes, n_gpus, buffer_size,
        is_origin_matrix=True, is_origin_image=True, is_duplex=False):

        # 参数赋值
        self.element_vector_path = element_vector_path

        self.image_x_size = image_x_size
        self.image_y_size = image_y_size
        self.max_objects = max_objects_per_image
        self.cell_x_size = cell_x_size
        self.cell_y_size = cell_y_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_channel = n_channel
        self.n_iters = int(n_iters / n_gpus)
        self.n_processes = n_processes
        self.buffer_size = buffer_size
        self.is_origin_matrix = is_origin_matrix
        self.is_origin_image = is_origin_image
        self.is_duplex = is_duplex
        self.n_coord = 8 if self.is_duplex else 4

        self.image_index_size = (self.batch_size)
        if self.is_origin_matrix:
            self.image_all_size = (self.batch_size, self.image_y_size, self.image_x_size, 1)
        else:
            self.image_all_size = (self.batch_size, self.image_y_size, self.image_x_size, self.n_channel)
        self.coord_true_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects, self.n_coord)
        self.coord_mask_size = (self.batch_size, self.cell_y_size, self.cell_x_size, self.max_objects)
        self.up_coord_true_size = (self.batch_size, self.max_objects, self.n_coord)
        self.up_coord_mask_size = (self.batch_size, self.max_objects)
        self.dataset_size = sum([numpy.prod(t) for t in [
            self.image_index_size, self.image_all_size, self.coord_true_size, self.coord_mask_size, 
            self.up_coord_true_size, self.up_coord_mask_size]])
        
        # 初始化词向量和内容字典
        self.elements_vector = self.load_elements_vector(self.element_vector_path)
        
    def init_datasets(self, mode, train_image_paths_file=None,
        test_image_paths_file=None, deploy_items=None): 
        """
        初始化数据集
        输入1：mode - 训练/验证/测试/应用
        输入2：train_image_paths_file - 训练数据图片存储的路径文件
        输入3：test_image_paths_file - 测试数据图片存储的路径文件
        输入4：deploy_items - 应用数据图片存储的内存指针
        """
        # 根据mode进行image_processor的初始化，并判断参数是否出错
        if mode == 'train':
            if train_image_paths_file is None or not os.path.exists(train_image_paths_file) or \
                test_image_paths_file is None or not os.path.exists(test_image_paths_file):
                raise('ERROR: wrong parameters in initialization!')
        elif mode == 'valid':
            if test_image_paths_file is None or not os.path.exists(test_image_paths_file):
                raise('ERROR: wrong parameters in initialization!')
        elif mode == 'test':
            if test_image_paths_file is None or not os.path.exists(test_image_paths_file):
                raise('ERROR: wrong parameters in initialization!')
        elif mode == 'deploy':
            if deploy_items is None:
                raise('ERROR: wrong parameters in initialization!')
        else:
            raise('ERROR: wrong mode in initialization!')

        if mode == 'train':
            self.load_datasets('train', image_paths_file=train_image_paths_file)
            self.load_datasets('valid', image_paths_file=test_image_paths_file)
            self.shared_memory = SharedMemory(self.buffer_size, self.dataset_size)
            print('finish apply shared memory ...')
            sys.stdout.flush()
        elif mode == 'valid':
            self.load_datasets('valid', image_paths_file=test_image_paths_file)
        elif mode == 'test':
            self.load_datasets('test', image_paths_file=test_image_paths_file)
        elif mode == 'deploy':
            self.load_datasets('deploy', items=deploy_items)
    
    def load_datasets(self, mode, image_paths_file=None, items=None):
        """
        读取数据集
        输入1：mode - 训练/验证/测试/应用
        输入2：image_paths_file - 数据图片存储的路径文件
        输入3：items - 数据的list，每一个元素是一个数据item
        """
        # 在训练/验证/测试/应用不同模式下读取datasets
        datasets = []
        
        # 判断参数是否出错
        if mode in ['train', 'valid', 'test']:
            if image_paths_file is None or not os.path.exists(image_paths_file):
                raise('ERROR: wrong parameters in load_datasets!')
        elif mode == 'deploy':
            if items is None:
                raise('ERROR: wrong parameters in load_datasets!')
        else:
            raise('ERROR: wrong mode in load_datasets!')

        if mode == 'train':
            image_infos = self.load_image_paths_from_file(image_paths_file)
            labels = self.load_labels_from_file(image_paths_file)
            if len(image_infos) != len(labels):
                raise('ERROR: image_paths and labels does not match!')
            
            # 组织datasets, datasets是一个item的list
            for image_info, label in zip(image_infos, labels):
                [image_path, image_name] = image_info
                [docid, pageid, tag] = image_name.split('_')
                item = {'docid': docid, 'pageid': pageid, 'tag': tag,
                    'path': image_path, 'label': label}
                datasets.append(item)
            
            self.trainsets = datasets
            self.n_train = len(self.trainsets)
            print('number of trainsets: %d' % (self.n_train))
            sys.stdout.flush()
        elif mode == 'valid':
            image_infos = self.load_image_paths_from_file(image_paths_file)
            labels = self.load_labels_from_file(image_paths_file)
            jsons = self.load_jsons_from_file(image_paths_file)
            if len(image_infos) != len(labels):
                raise('ERROR: image_paths and labels does not match!')
            
            # 组织datasets, datasets是一个item的list
            if len(image_infos) != len(labels):
                raise('ERROR: image_paths and labels does not match!')
            for image_info, label, json in zip(image_infos, labels, jsons):
                [image_path, image_name] = image_info
                image = cv2.imread(image_path, -1)
                [docid, pageid, tag] = image_name.split('_')
                item = {'docid': docid, 'pageid': pageid, 'tag': tag,
                    'path': image_path, 'label': label,
                    'orig_image': image, 'content': json}
                datasets.append(item)
            
            self.validsets = datasets
            self.n_valid = len(self.validsets)
            print('number of validsets: %d' % (self.n_valid))
            sys.stdout.flush()
        elif mode == 'test':
            image_infos = self.load_image_paths_from_file(image_paths_file)
            jsons = self.load_jsons_from_file(image_paths_file)
            
            # 组织datasets, datasets是一个item的list
            for image_info, json in zip(image_infos, jsons):
                [image_path, image_name] = image_info
                image = cv2.imread(image_path, -1)
                [docid, pageid, tag] = image_name.split('_')
                item = {'docid': docid, 'pageid': pageid, 'tag': tag,
                    'path': image_path, 'orig_image': image, 'content': json}
                datasets.append(item)
            
            self.testsets = datasets
            self.n_test = len(self.testsets)
            print('number of testsets: %d' % (self.n_test))
            sys.stdout.flush()
        elif mode == 'deploy':
            # 组织datasets, datasets是一个item的list
            self.deploysets = items
            self.n_deploy = len(self.deploysets)
            print('number of deploysets: %d' % (self.n_deploy))
            sys.stdout.flush()

    def load_image_paths_from_file(self, image_paths_file):
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
                    raise('ERROR: image path not exists!')
                file_name = os.path.split(image_path)[1]
                image_name = os.path.splitext(file_name)[0]
                image_paths.append([image_path, image_name])

        return image_paths

    def load_labels_from_file(self, image_paths_file):
        """
        从文件中读取所有图片对应的标签的路径
        输入：image_paths_file - 图片路径文件
        输出：labels - 图片标签list，每一个元素是一张图片对应的标签
        """
        label_paths = []
        labels = []
        
        with open(image_paths_file, 'r') as fo:
            for line in fo:
                image_path = line.strip()
                label_path = image_path.replace('tensors', 'labels')
                label_path = label_path.replace('.png', '.txt')
                if not os.path.exists(label_path):
                    raise('ERROR: label path not exists!')
                label_paths.append(label_path)

        for label_path in label_paths:
            label = [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * self.max_objects
            n_objects = 0
            
            # 读取标签路径
            with open(label_path, 'r') as fo:
                for line in fo:
                    infos = line.strip().split(' ')
                
                    index = float(infos[0])
                    in_x = float(infos[1])
                    in_y = float(infos[2])
                    in_w = float(infos[3])
                    in_h = float(infos[4])
                    out_x = float(infos[5])
                    out_y = float(infos[6])
                    out_w = float(infos[7])
                    out_h = float(infos[8])
                
                    in_left = in_x - in_w / 2.0
                    in_right = in_x + in_w / 2.0
                    in_top = in_y - in_h / 2.0
                    in_bottom = in_y + in_h / 2.0
                    out_left = out_x - out_w / 2.0
                    out_right = out_x + out_w / 2.0
                    out_top = out_y - out_h / 2.0
                    out_bottom = out_y + out_h / 2.0
                    
                    label[n_objects] = [index, in_left, in_right, in_top, in_bottom,
                        out_left, out_right, out_top, out_bottom]
                    n_objects += 1
                    if n_objects >= self.max_objects:
                        break

            labels.append(label)
        
        return labels

    def load_jsons_from_file(self, image_paths_file):
        """
        从文件中读取所有图片对应的json文件的路径
        输入：image_paths_file - 图片路径文件
        输出：jsons - 图片内容dict
        """
        jsons = []
        
        with open(image_paths_file, 'r') as fo:
            for line in fo:
                image_path = line.strip()
                json_path = image_path.replace('tensors', 'jsons')
                json_path = json_path.replace('.png', '.json')
                if not os.path.exists(json_path):
                    raise('ERROR: json_path not exists!')
                
                with open(json_path, 'r') as fo:
                    data = json.load(fo)
                jsons.append(data)

        return jsons
    
    def load_elements_vector(self, path):
        """
        读取词向量矩阵
        输入：path - 词向量文件
        输出：elements_vector - 词向量矩阵，numpy类型，每一行是一个词向量
        """
        elements_vector = []

        if path != None and os.path.exists(path):
            with open(path, 'r') as fo:
                for line in fo:
                    vector = [float(t) for t in line.strip().split('\t')[1].split(' ')]
                    if len(vector) != self.n_channel:
                        raise('elements vectors do not match n_channel')
                    elements_vector.append(vector)

        print('number of elements vector: %d' % (len(elements_vector)))
        sys.stdout.flush()
        
        return numpy.array(elements_vector, dtype='float32')
    
    def load_content_dict(self, path):
        """
        读取图片内容
        输入：path - 验证集和测试集图片的内容字典
        输出：content_dict - 内容字典，key是文档id，valud是文档对应的内容字典
        """
        content_dict = {}

        if path != None and os.path.exists(path):
            doc_paths = []
            with open(path, 'r') as fo:
                for line in fo:
                    doc_path = line.strip()
                    if os.path.exists(doc_path):
                        doc_paths.append(doc_path)

            for doc_path in doc_paths:
                [directory, file_name] = os.path.split(doc_path)
                docid = os.path.split(directory)[1]
                with open(doc_path, 'r') as fo:
                    content_dict[docid] = json.load(fo)
        
        print('number of content dict: %d' % (len(content_dict)))
        sys.stdout.flush()

        return content_dict

    def dataset_producer_based_shm(self):
        """
        基于共享内存的方法生产数据
        """
        while True:
            batch_image_indexs, batch_image_paths, batch_labels = \
                self.get_random_batch(self.trainsets, self.batch_size)
            batch_image_indexs = numpy.array(batch_image_indexs, dtype='float32')
            batch_images = self.get_images_from_paths(batch_image_paths)
            batch_images = self.convert_batch_images(batch_images)
            # 数据增强
            batch_images, batch_labels = self.data_augmentation(
                mode='train', batch_images=batch_images, batch_labels=batch_labels)
            batch_coord_true, batch_object_mask, \
                batch_unpositioned_coord_true, batch_unpositioned_object_mask = \
                self.convert_batch_labels(batch_labels)
            # 改变尺寸存入shared_memory
            batch_image_indexs = batch_image_indexs.flatten()
            batch_images = batch_images.flatten()
            batch_coord_true = batch_coord_true.flatten()
            batch_object_mask = batch_object_mask.flatten()
            batch_up_coord_true = batch_unpositioned_coord_true.flatten()
            batch_up_object_mask = batch_unpositioned_object_mask.flatten()
            dataset = numpy.concatenate([
                batch_image_indexs, batch_images, batch_coord_true, batch_object_mask, 
                batch_up_coord_true, batch_up_object_mask], axis=0)
            
            self.shared_memory.put(dataset)
    
    def get_random_batch(self, dataset, batch_size):
        """
        获取随机一个batch的数据
        输入1：dataset - 整个数据集
        输入2：batch_size - 该batch的大小
        输出1：batch_image_indexs - 每条数据在整个数据集中的index
        输出2：batch_image_paths - 每条数据的图片路径
        输出3：batch_labels - 每条数据的标签
        """
        batch_image_indexs, batch_image_paths, batch_labels = [], [], []
       
        for i in range(batch_size):
            index = random.randint(0, len(dataset)-1)
            item = dataset[index]
            image_path = item['path']
            label = item['label']
            batch_image_indexs.append(index)
            batch_image_paths.append(image_path)
            batch_labels.append(label)
            
        return batch_image_indexs, batch_image_paths, batch_labels

    def get_images_from_paths(self, batch_image_paths):
        # 从路径中读取图片
        batch_images = []
        for image_path in batch_image_paths:
            image = cv2.imread(image_path, -1)
            batch_images.append(image)

        return batch_images

    def dataset_producer(self, mode, index, batch_size):
        # 直接从内存获取一个batch的数据 
        if mode == 'valid':
            batch_image_paths, batch_labels = [], []
            for j in range(batch_size):
                item = self.validsets[index+j]
                batch_image_paths.append(item['path'])
                batch_labels.append(item['label'])
            batch_images = self.get_images_from_paths(batch_image_paths)
            batch_images = self.convert_batch_images(batch_images)
        elif mode == 'test':
            batch_image_paths, batch_labels = [], None
            for j in range(batch_size):
                item = self.testsets[index+j]
                batch_image_paths.append(item['path'])
            batch_images = self.get_images_from_paths(batch_image_paths)
            batch_images = self.convert_batch_images(batch_images)
        elif mode == 'deploy':
            batch_images, batch_labels = [], None
            for j in range(batch_size):
                item = self.deploysets[index+j]
                batch_images.append(item['orig_image'])
            batch_images = self.convert_batch_images(batch_images)
        
        batch_images = self.data_augmentation(mode='test', batch_images=batch_images)

        return batch_images

    def convert_batch_images(self, images):
        # 转化图片，按照模型的不同，改变图片的第三维度，返回image的list
        new_images = []
        
        for image in images:
            h, w = image.shape[0], image.shape[1]
            if self.is_origin_matrix:
                image = numpy.reshape(image, newshape=(h, w, 1))
            elif self.is_origin_image:
                image = numpy.reshape(image/255.0, newshape=(h, w, self.n_channel))
            else:
                image = self.elements_vector[image.flatten()].reshape(
                    [h, w, self.n_channel])
            new_images.append(image)

        return new_images

    def data_augmentation(self, mode, batch_images=None, batch_labels=None):
        # 数据增强，返回的images和labels都是numpy类型
        # 判断参数是否出错
        if mode == 'train':
            if batch_images is None or batch_labels is None:
                raise('ERROR: wrong parameters in data_augmentation!')
        elif mode == 'test':
            if batch_images is None:
                raise('ERROR: wrong parameters in data_augmentation!')
        else:
            raise('ERROR: wrong mode in data_augmentation!')
        
        # 图像尺寸变换
        if mode == 'train':
            new_images, new_labels = [], []
            # 训练时，random resize + flip + adjust label
            for image, label in zip(batch_images, batch_labels):
                image, label = self.image_random_resize(
                    image, label, jitter=0.2)
                image, label = self.image_flip(image, label)
                new_images.append(image)
                new_labels.append(label)
            new_labels = numpy.array(new_labels, dtype='float32')
        elif mode == 'test':
            # 测试时，fixed resize
            new_images = []
            for image in batch_images:
                image = self.image_fixed_resize(image)
                new_images.append(image)
         
        if self.is_origin_matrix:
            new_images = numpy.array(new_images, dtype='int32')
        else:
            new_images = numpy.array(new_images, dtype='float16')

        if mode == 'train':
            return new_images, new_labels
        elif mode == 'test':
            return new_images

    def image_random_resize(self, image, label, jitter=0.2):
        resized_w, resized_h = int(self.image_x_size), int(self.image_y_size)
        orig_w, orig_h, n_channel = image.shape[1], image.shape[0], image.shape[2]
        
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

        # temp_image = cv2.resize(image, dsize=(nw, nh))
        temp_image = self.resize(image, orig_w, orig_h, nw, nh)
        temp_image = numpy.reshape(temp_image, (nh, nw, n_channel))
        
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
        
        new_image = numpy.zeros(shape=(resized_h, resized_w, n_channel))
        new_image[new_sy: new_ey, new_sx: new_ex, :] = \
            temp_image[old_sy: old_ey, old_sx: old_ex, :]

        # 重新计算labels
        new_label = [[0, 0, 0, 0, 0, 0, 0, 0, 0]] * self.max_objects
        n = 0
        
        for j in range(len(label)):
            if sum(label[j]) == 0:
                break
                    
            [index, in_left, in_right, in_top, in_bottom, 
                out_left, out_right, out_top, out_bottom] = label[j]
            
            # 按照nw，nh的比例重新定位内框
            if resized_w > nw:
                new_in_left = (1.0 * in_left * nw + dx) / resized_w
                new_in_right = (1.0 * in_right * nw + dx) / resized_w
            else:
                new_in_left = (1.0 * in_left * nw - dx) / resized_w
                new_in_right = (1.0 * in_right * nw - dx) / resized_w
                
            if resized_h > nh:
                new_in_top = (1.0 * in_top * nh + dy) / resized_h
                new_in_bottom = (1.0 * in_bottom * nh + dy) / resized_h
            else:
                new_in_top = (1.0 * in_top * nh - dy) / resized_h
                new_in_bottom = (1.0 * in_bottom * nh - dy) / resized_h
                
            new_in_left = min(max(0.0, new_in_left), 1.0 - 1e-6)
            new_in_right = max(0.0, min(new_in_right, 1.0 - 1e-6))
            new_in_top = min(max(0.0, new_in_top), 1.0 - 1e-6)
            new_in_bottom = max(0.0, min(new_in_bottom, 1.0 - 1e-6))
            
            # 按照nw，nh的比例重新定位外框
            if resized_w > nw:
                new_out_left = (1.0 * out_left * nw + dx) / resized_w
                new_out_right = (1.0 * out_right * nw + dx) / resized_w
            else:
                new_out_left = (1.0 * out_left * nw - dx) / resized_w
                new_out_right = (1.0 * out_right * nw - dx) / resized_w
                
            if resized_h > nh:
                new_out_top = (1.0 * out_top * nh + dy) / resized_h
                new_out_bottom = (1.0 * out_bottom * nh + dy) / resized_h
            else:
                new_out_top = (1.0 * out_top * nh - dy) / resized_h
                new_out_bottom = (1.0 * out_bottom * nh - dy) / resized_h
                
            new_out_left = min(max(0.0, new_out_left), 1.0 - 1e-6)
            new_out_right = max(0.0, min(new_out_right, 1.0 - 1e-6))
            new_out_top = min(max(0.0, new_out_top), 1.0 - 1e-6)
            new_out_bottom = max(0.0, min(new_out_bottom, 1.0 - 1e-6))
            
            if new_in_right > new_in_left and new_in_bottom > new_in_top and \
                new_out_right > new_out_left and new_out_bottom > new_out_top:
                new_label[n] = [index, new_in_left, new_in_right, new_in_top, new_in_bottom,
                    new_out_left, new_out_right, new_out_top, new_out_bottom]
                n += 1

        return new_image, new_label

    def image_fixed_resize(self, image):
        resized_w, resized_h = int(self.image_x_size), int(self.image_y_size)
        orig_w, orig_h, n_channel = image.shape[1], image.shape[0], image.shape[2]
        
        new_w, new_h = orig_w, orig_h

        if 1.0 * resized_w / orig_w  < 1.0 * resized_h / orig_h:
            new_w = int(resized_w)
            new_h = int(1.0 * resized_w * orig_h / orig_w)
        else:
            new_h = int(resized_h)
            new_w = int(1.0 * resized_h * orig_w / orig_h)
       
        # temp_image = cv2.resize(image, dsize=(new_w, new_h))
        temp_image = self.resize(image, orig_w, orig_h, new_w, new_h)
        temp_image = numpy.reshape(temp_image, (new_h, new_w, n_channel))
        
        dx = int((resized_w - new_w) / 2.0)
        dy = int((resized_h - new_h) / 2.0)
        
        sx, ex = dx, dx + new_w
        sy, ey = dy, dy + new_h
        
        new_image = numpy.zeros(shape=(resized_h, resized_w, n_channel))
        new_image[sy:ey, sx:ex, :] = temp_image
        
        return new_image
    
    def image_flip(self, image, label):
        # 图像翻转
        old_image = image
        new_label = label
        orig_w, orig_h, n_channel = image.shape[1], image.shape[0], image.shape[2]
        if numpy.random.random() < 0.5:
            new_image = cv2.flip(old_image, 1)
            new_image = numpy.reshape(new_image, (orig_h, orig_w, n_channel))
                
            # 重新计算box label
            for j in range(len(label)):
                if sum(label[j]) == 0:
                    break
                in_right = 1.0 - new_label[j][1]
                in_left = 1.0 - new_label[j][2]
                out_right = 1.0 - new_label[j][5]
                out_left = 1.0 - new_label[j][6]
                new_label[j][1] = in_left
                new_label[j][2] = in_right
                new_label[j][5] = out_left
                new_label[j][6] = out_right
        else:
            new_image = old_image
            
        return new_image, new_label

    def resize(self, image, old_w, old_h, new_w, new_h):
        cols = numpy.array(numpy.array(range(new_h)) * old_h / new_h, dtype='int32')
        cols = numpy.reshape(cols, (new_h, 1))
        cols = numpy.tile(cols, (1, new_w))
        rows = numpy.array(numpy.array(range(new_w)) * old_w / new_w, dtype='int32')
        rows = numpy.reshape(rows, (1, new_w))
        rows = numpy.tile(rows, (new_h, 1))
        new_image = image[cols, rows]
        new_image = numpy.reshape(new_image, (new_h, new_w, image.shape[2]))

        return new_image
    
    def convert_batch_labels(self, batch_labels):
        batch_coord_true, batch_object_mask, \
            batch_unpositioned_coord_true, batch_unpositioned_object_mask = [], [], [], []
            
        for i in range(len(batch_labels)):
            coord_true, object_mask, unpositioned_coord_true, unpositioned_object_mask = \
                self.process_label(batch_labels[i])
            batch_coord_true.append(coord_true)
            batch_object_mask.append(object_mask)
            batch_unpositioned_coord_true.append(unpositioned_coord_true)
            batch_unpositioned_object_mask.append(unpositioned_object_mask)
        
        batch_coord_true = numpy.array(batch_coord_true, dtype='float32')
        batch_object_mask = numpy.array(batch_object_mask, dtype='float32')
        batch_unpositioned_coord_true = numpy.array(batch_unpositioned_coord_true, dtype='float32')
        batch_unpositioned_object_mask = numpy.array(batch_unpositioned_object_mask, dtype='float32')
        batch_coord_true = batch_coord_true[:,:,:,:,0:self.n_coord]
        batch_unpositioned_coord_true = batch_unpositioned_coord_true[:,:,0:self.n_coord]
        
        return batch_coord_true, batch_object_mask, \
            batch_unpositioned_coord_true, batch_unpositioned_object_mask

    def process_label(self, label):        
        # true_label and mask in 包围框标记
        coord_true = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size, self.max_objects, 8),
            dtype='float32')
        object_mask = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size, self.max_objects),
            dtype='float32')
        unpositioned_coord_true = numpy.zeros(
            shape=(self.max_objects, 8),
            dtype='float32')
        unpositioned_object_mask = numpy.zeros(
            shape=(self.max_objects, ),
            dtype='float32')
        object_nums = numpy.zeros(
            shape=(self.cell_y_size, self.cell_x_size),
            dtype='int')
        
        for j in range(self.max_objects):
            
            [index, in_left, in_right, in_top, in_bottom, 
                out_left, out_right, out_top, out_bottom] = label[j]
            
            index = int(index)
            center_x = (in_left + in_right) / 2.0
            center_y = (in_top + in_bottom) / 2.0
            w = in_right - in_left
            h = in_bottom - in_top
            
            if not ((in_left == 0.0 and in_right == 0.0 and in_top == 0.0 and in_bottom == 0.0) or \
                (out_left == 0.0 and out_right == 0.0 and out_top == 0.0 and out_bottom == 0.0)):
                # 计算包围框标记
                center_cell_x = int(math.floor(self.cell_x_size * center_x))
                center_cell_y = int(math.floor(self.cell_y_size * center_y))
                coord_true[center_cell_y, center_cell_x, 
                    object_nums[center_cell_y, center_cell_x]] = \
                        numpy.array([center_x, center_y, w, h, out_left, out_right, out_top, out_bottom])
                object_mask[center_cell_y, center_cell_x, 
                    object_nums[center_cell_y, center_cell_x]] = 1.0
                object_nums[center_cell_y, center_cell_x] += 1

                unpositioned_coord_true[j,:] = numpy.array([
                    center_x, center_y, w, h, out_left, out_right, out_top, out_bottom])
                unpositioned_object_mask[j] = 1.0
                            
        return coord_true, object_mask, unpositioned_coord_true, unpositioned_object_mask


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

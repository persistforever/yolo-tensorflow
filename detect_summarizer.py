# -*- coding: utf8 -*-
# author: ronniecao
# time: 2018/03/10
# description: start script
from __future__ import print_function
import sys
import argparse
import os
import platform
import collections
import random
import numpy
import multiprocessing as mp
from multiprocessing.sharedctypes import Array
from ctypes import c_double, cast, POINTER
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

if 'Windows' in platform.platform():
    store_dir = 'E:\Github\\table-detection\\'
elif 'Linux' in platform.platform():
    data_dir = '/home/caory/github/yolo-tensorflow/'


def main(method='train', gpus=''):
    from src.data.data_summarizer import Processor
    from src.network.network_summarizer import Network
    from src.model.model_summarizer import Model
   
    option = collections.OrderedDict()
    option['batch_size'] = 32
    option['image_x_size'] = 448
    option['image_y_size'] = 448
    option['n_channel'] = 3
    option['n_classes'] = 21
    option['cell_x_size'] = 7
    option['cell_y_size'] = 7
    option['pool_mode'] = 'max'
    option['n_boxes'] = 5
    option['n_processes'] = 2
    option['max_objects'] = 30
    option['n_iter'] = 200000
    option['buffer_size'] = 5
    option['gpus'] = gpus
    option['n_gpus'] = len(gpus.split(',')) if len(gpus.split(',')) != 0 else 1
    option['is_multigpu'] = True if option['n_gpus'] > 1 else False
    option['is_valid'] = False
    option['is_observe'] = False
    option['noobject_scale'] = 1
    option['object_scale'] = 1
    option['coord_scale'] = 1
    option['class_scale'] = 1
    option['is_weight_decay'] = False
    option['weight_decay'] = 1e-3
    option['learning_rate'] = 1e-4
    option['is_lr_decay'] = False
    option['train_data'] = 'voc'
    option['test_data'] = 'voc'
    option['seq'] = 'voc-v1'
    option['model'] = 'model_best.ckpt'
    option['update_function'] = 'momentum'
    
    # 打印option
    print()
    for key in option:
        print('%-20s' % (key), '= {}'.format(option[key]))
    print()
    
    processor = Processor(
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'], 
        max_objects = option['max_objects'], 
        n_classes = option['n_classes'],
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'],
        batch_size = option['batch_size'], 
        n_channel = option['n_channel'],
        n_processes = option['n_processes'], 
        n_iters = option['n_iter'], 
        buffer_size = option['buffer_size'])
        
    network = Network(
        n_channel = option['n_channel'], 
        n_classes = option['n_classes'], 
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'],
        max_objects = option['max_objects'], 
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'], 
        pool_mode = option['pool_mode'],
        box_per_cell = option['n_boxes'], 
        batch_size = option['batch_size'],
        object_scale = option['object_scale'], 
        noobject_scale = option['noobject_scale'], 
        coord_scale = option['coord_scale'], 
        class_scale = option['class_scale'],
        noobject_thresh = 0.6, 
        recall_thresh = 0.6, 
        pred_thresh = 0.5, 
        nms_thresh = 0.4,
        is_weight_decay = option['is_weight_decay'],
        weight_decay_scale = option['weight_decay'])
    
    model = Model(
        n_channel = option['n_channel'], 
        max_objects = option['max_objects'],
        image_x_size = option['image_x_size'], 
        image_y_size = option['image_y_size'], 
        cell_x_size = option['cell_x_size'], 
        cell_y_size = option['cell_y_size'],
        n_classes = option['n_classes'],
        box_per_cell = option['n_boxes'], 
        batch_size = option['batch_size'],
        buffer_size = option['buffer_size'],
        is_valid = option['is_valid'], 
        update_function = option['update_function'], 
        learning_rate = option['learning_rate'],
        is_lr_decay = option['is_lr_decay']) 
    
    if method == 'train':
        # 训练模型
        train_image_paths_file = os.path.join(data_dir, 'datasets', option['train_data'], 'train.txt')
        test_image_paths_file = os.path.join(data_dir, 'datasets', option['test_data'], 'valid.txt')
        processor.init_datasets(mode='train', 
            train_image_paths_file=train_image_paths_file, 
            test_image_paths_file=test_image_paths_file)
        
        # 设置数据池，processor负责生产dataset，model负责消费dataset
        producers = []
        for i in range(option['n_processes']):
            producer = mp.Process(
                target=processor.dataset_producer_based_shm, args=(i,), name='producer%d' % (i))
            producers.append(producer)
        # 在CPU中运行生产者
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for producer in producers:
            producer.start()
        # 在GPU中运行消费者
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        model.train(
            processor, network, 
            backup_dir=os.path.join(data_dir, 'backup', option['seq']), 
            logs_dir=os.path.join(data_dir, 'logs', option['seq']), 
            n_iters=option['n_iter'])
        
    elif method == 'test':
        # 测试某一个已经训练好的模型
        test_image_paths_files=[os.path.join(data_dir, 'datasets', option['datas'][-1], 'test_tensor.txt')]
        processor.init_datasets(mode='test', test_image_paths_files=test_image_paths_files)
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        model_path = os.path.join(store_dir, 'backup', option['seq'], option['sub_dir'], option['model'])
        model.test_model(
            processor=processor, network=network, model_path=model_path,
            output_dir=os.path.join(store_dir, 'logs', option['seq']))


if __name__ == '__main__':
    print('current process id: %d' % (os.getpid()))
    parser = argparse.ArgumentParser(description='parsing command parameters')
    parser.add_argument('-method')
    parser.add_argument('-gpus')
    parser.add_argument('-name')
    arg = parser.parse_args()
    method = arg.method
    gpus = arg.gpus if arg.gpus else ''
    main(method=method, gpus=gpus)

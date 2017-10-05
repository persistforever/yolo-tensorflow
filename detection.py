# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import platform

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

if 'Windows' in platform.platform():
    maindir = 'E:\Github\\table-detection\\'
elif 'Linux' in platform.platform():
    datadir = '/home/caory/github/yolo-tensorflow/'
    storedir = '/home/caory/github/yolo-tensorflow/'


def train():
    from src.data.image_mp import ImageProcessor
    from src.model.yolo_v1 import TinyYolo
    
    image_processor = ImageProcessor(
        os.path.join(datadir, 'datasets', 'voc-v2'),
        image_size=448, max_objects_per_image=30, cell_size=7, n_classes=1,
        n_processes=2, batch_size=64)
    
    tiny_yolo = TinyYolo(
        n_channel=3, n_classes=20, image_size=448, max_objects_per_image=30,
        cell_size=7, box_per_cell=5, object_scale=1, noobject_scale=0.5,
        coord_scale=5, class_scale=1, batch_size=64, noobject_thresh=0.6,
        recall_thresh=0.5, pred_thresh=0.5, nms_thresh=0.4)
    
    tiny_yolo.train(
        processor=image_processor, backup_path=os.path.join(storedir, 'backup', 'voc-v2'),
        n_iters=500000, batch_size=64)
    
def test():
    from src.data.image import ImageProcessor
    from src.model.yolo_v1 import TinyYolo
    
    image_processor = ImageProcessor(
        os.path.join(datadir, 'datasets', 'voc-v2'),
        image_size=448, max_objects_per_image=30, cell_size=7, n_classes=1,
        n_processes=1, batch_size=64)
    
    tiny_yolo = TinyYolo(
        n_channel=3, n_classes=20, image_size=448, max_objects_per_image=30,
        cell_size=7, box_per_cell=5, object_scale=1, noobject_scale=0.5,
        coord_scale=5, class_scale=1, batch_size=64, noobject_thresh=0.6,
        recall_thresh=0.5, pred_thresh=0.5, nms_thresh=0.4)
    
    tiny_yolo.test(
        processor=image_processor, backup_dir=os.path.join(storedir, 'backup', 'voc-v2'),
        output_dir=os.path.join(storedir, 'logs', 'voc-v2', 'predictions'), batch_size=64)
    

train()

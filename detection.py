# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import platform

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = 'CPU'

if 'Windows' in platform.platform():
    maindir = 'E:\Github\\table-detection\\'
elif 'Linux' in platform.platform():
    datadir = '/home/caory/github/darknet-table/'
    storedir = '/home/caory/github/table-detection/'


def train():
    from src.data.image import ImageProcessor
    from src.model.yolo_v1 import TinyYolo
    
    image_processor = ImageProcessor(
        os.path.join(datadir, 'datasets', 'table-v1'),
        image_size=448, max_objects_per_image=30, cell_size=4, n_classes=1)
    
    tiny_yolo = TinyYolo(
        n_channel=3, n_classes=1, image_size=448, max_objects_per_image=30,
        cell_size=4, box_per_cell=7, object_scale=10, noobject_scale=3,
        coord_scale=10, batch_size=64, noobject_thresh=0.6,
        recall_thresh=0.5)
    
    tiny_yolo.train(
        processor=image_processor, backup_path=os.path.join(storedir, 'backup', 'table-v1'),
        n_iters=100000, batch_size=128)
    
    
def test():
    from src.data.image import ImageProcessor
    from src.model.yolo_v1 import TinyYolo
    
    image_processor = ImageProcessor(
        os.path.join(maindir, 'data', 'table-v1'),
        image_size=256, max_objects_per_image=30, cell_size=4, n_classes=1)
    
    tiny_yolo = TinyYolo(
        n_channel=3, n_classes=1, image_size=256, max_objects_per_image=30,
        cell_size=4, box_per_cell=5, object_scale=10, noobject_scale=3,
        coord_scale=10, batch_size=1, noobject_thresh=0.6,
        recall_thresh=0.5)
    
    tiny_yolo.test(
        processor=image_processor, backup_path=os.path.join(maindir, 'backup', 'table-v3'),
        batch_size=1)
    
    
test()

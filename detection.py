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
    maindir = '/home/caory/github/table-detection/'


def train():
    from src.data.image import ImageProcessor
    from src.model.yolo_v2 import TinyYolo
    
    tiny_yolo = TinyYolo(
        n_channel=3, n_classes=1, image_size=224, max_objects_per_image=30,
        cell_size=7, box_per_cell=6, object_scala=1, nobject_scala=0.5,
        coord_scala=5, class_scala=1, batch_size=64, nobject_thresh=0.6,
        recall_thresh=0.5)
    
    sys.stdout.flush()
    image_processor = ImageProcessor(
        os.path.join(maindir, 'data', 'table-v1'),
        image_size=224, max_objects_per_image=30, cell_size=7, n_classes=1)
    print('Processing Images finished!\n')
    sys.stdout.flush()
    
    tiny_yolo.train(
        processor=image_processor, backup_path=os.path.join(maindir, 'backup', 'table-v5'),
        n_iters=500000, batch_size=64)
    
    
def uint_test():
    from src.data.test.image_test import ImageProcessorTestor
    
    testor = ImageProcessorTestor()
    testor.test_image_resize()
    
    
train()

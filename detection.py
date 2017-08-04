# -*- coding: utf8 -*-
# author: ronniecao
import os
from src.data.image import ImageProcessor
from src.model.tiny_yolo import TinyYolo

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

image_processor = ImageProcessor(
    'E:\Github\\table-detection\\data\\table-v1', image_size=288, max_objects_per_image=20)
tiny_yolo = TinyYolo(
    n_channel=3, n_classes=1, image_size=384, max_objects_per_image=20, 
    cell_size=7, box_per_cell=5)
tiny_yolo.train(
    processor=image_processor, backup_path='E:\Github\\table-detection\\backup\\table-v1',
    n_epoch=5, batch_size=2)
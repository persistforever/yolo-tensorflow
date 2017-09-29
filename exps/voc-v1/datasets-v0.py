# -*- coding: utf-8 -*-
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2007', 'train')]

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(in_file, out_file):
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        with open(out_file, 'w') as fw:
            fw.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def construct_label(source_dir, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    if not os.path.exists(os.path.join(target_dir, 'Labels')):
        os.makedirs(os.path.join(target_dir, 'Labels'))
    xml_list = os.listdir(os.path.join(source_dir, 'Annotations'))
    for xmlname in xml_list:
        xmlpath = os.path.join(source_dir, 'Annotations', xmlname)
        outpath = os.path.join(target_dir, 'Labels', xmlname.split('.')[0]+'.txt')
        convert_annotation(xmlpath, outpath)

def construct_dataset(source_dir, target_dir):
    trainsets, testsets = [], []
    with open(os.path.join(source_dir, 'ImageSets', 'Main', 'cat_train.txt'), 'r') as fo:
        for line in fo:
            filename = line.strip().split(' ')[0]
            filepath = os.path.join(source_dir, 'Images', '%s.jpg' % (filename))
            trainsets.append(filepath)

    with open(os.path.join(source_dir, 'ImageSets', 'Main', 'cat_val.txt'), 'r') as fo:
        for line in fo:
            filename = line.strip().split(' ')[0]
            filepath = os.path.join(source_dir, 'Images', '%s.jpg' % (filename))
            testsets.append(filepath)

    with open(os.path.join(target_dir, 'train.txt'), 'w') as fw:
        for filepath in trainsets:
            fw.writelines(('%s\n' % (filepath)).encode('utf8'))

    with open(os.path.join(target_dir, 'test.txt'), 'w') as fw:
        for filepath in testsets:
            fw.writelines(('%s\n' % (filepath)).encode('utf8'))

construct_label('/home/caory/github/yolo-tensorflow/datasets/voc-v0/VOC2007', '/home/caory/github/yolo-tensorflow/datasets/voc-v2')
construct_dataset('/home/caory/github/yolo-tensorflow/datasets/voc-v0/VOC2012', '/home/caory/github/yolo-tensorflow/datasets/voc-v2')

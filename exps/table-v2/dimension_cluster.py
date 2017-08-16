# -*- coding: utf8 -*-
# author: ronniecao
from __future__ import print_function
import sys
import os
import platform
import cv2
import numpy
import matplotlib.pyplot as plt


if 'Windows' in platform.platform():
    maindir = 'E:\Github\\table-detection\\'
elif 'Linux' in platform.platform():
    maindir = '/home/caory/github/table-detection/'


def read_box(maindir):
    path = os.path.join(maindir, 'data', 'table-v1', 'train.txt')
    box_lists = []
    n_objects = 0
    with open(path, 'r') as fo:
        for line in fo:
            infos = line.strip().split(' ')
            n_objects += 1
            if n_objects % 1000 == 0:
                print(n_objects)

            image_path = os.path.join(maindir, infos[0])
            label_infos = infos[1:]
                
            # 读取图像
            image = cv2.imread(image_path)
            [image_h, image_w, _] = image.shape
            
            # 处理 label
            i = 0
            while i < len(label_infos):
                xmin = int(label_infos[i])
                ymin = int(label_infos[i+1])
                xmax = int(label_infos[i+2])
                ymax = int(label_infos[i+3])
                class_index = int(label_infos[i+4])
                
                # 转化成 center_x, center_y, w, h
                xmin = 1.0 * xmin / image_w
                ymin = 1.0 * ymin / image_h
                xmax = 1.0 * xmax / image_w
                ymax = 1.0 * ymax / image_h
                box_lists.append([xmin, ymin, xmax, ymax])
                
                i += 5
        
        return numpy.array(box_lists)

def draw_centroids(centroids):
    # 画图
    image = numpy.zeros(shape=(100, 100, 3), dtype='uint8') + 255
    for i in range(centroids.shape[0]):
        [xmin, ymin, xmax, ymax] = centroids[i,:]
        xmin = int(xmin * 100)
        ymin = int(ymin * 100)
        xmax = int(xmax * 100)
        ymax = int(ymax * 100)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 99, 71), 1)
    plt.imshow(image)
    plt.show()

def cal_iou(box1, box2):
    left_top = numpy.maximum(box1[0:2], box2[0:2])
    right_bottom = numpy.minimum(box1[2:4], box2[2:4])
    intersection = right_bottom - left_top
    inter_area = intersection[0] * intersection[1]
    inter_area = inter_area if inter_area > 0 else 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area + 1e-6)

    return iou

def clustering(box_lists, n_centroid=5, n_iters=1000):
    # 初始化质心
    centroids = numpy.random.random(size=(n_centroid, 4))
    for j in range(n_centroid):
        [x, y, w, h] = [0.5, 1.0 * j / n_centroid + 1.0 / n_centroid / 2.0, 0.8, 1.0 / n_centroid]
        xmin = max(0.0, x - w / 2.0)
        xmax = min(x + w / 2.0, 1.0)
        ymin = max(0.0, y - h / 2.0)
        ymax = min(y + h / 2.0, 1.0)
        centroids[j,:] = numpy.array([xmin, ymin, xmax, ymax])

    for n in range(n_iters):
        out_str = '['
        for i in range(n_centroid):
            out_str += '[' + ','.join(['%.2f' % (t) for t in centroids[i,:]]) + '],'
        out_str += ']'
        print(out_str)
        
        classes = []
        for i in range(n_centroid):
            classes.append([])

        iou_value = 0.0
        # 第一步：计算每个example到centroid的距离，并且分类到距离最近的centroid。
        for i in range(box_lists.shape[0]):
            box = box_lists[i,:]
            distance = []
            for j in range(n_centroid):
                centroid = centroids[j,:]
                iou = cal_iou(centroid, box)
                distance.append(1.0 - iou)
            [label, dis] = min(enumerate(distance), key=lambda x: x[1])
            iou_value += (1-dis)
            classes[label].append(i)
        print('n_centroid: %d, iou: %.4f' % (n_centroid, iou_value / box_lists.shape[0]))

        # 第二步：重新计算每个类别的中心。
        for i in range(n_centroid):
            points = box_lists[classes[i]]
            if points.shape[0] > 0:
                centroids[i,:] = numpy.mean(points, axis=0)

def plot_curve():
    ious = [0.1906, 0.3357, 0.4385, 0.4948, 0.5223, 0.5687, 0.5924, 0.6108, 0.62260, 0.6333, 0.6482, 0.6563, 0.6661, 0.6659, 0.6791]

    fig = plt.figure(figsize=(10, 8))

    p1 = plt.plot(range(1,16), ious, 'o-', color='#66CDAA', markersize=10.0, linewidth=2.0)
    plt.grid(True)
    plt.title('Dimension Clusters')
    plt.xlabel('# of clusters')
    plt.ylabel('average IOU')
    # plt.show()
    plt.savefig('E:\\Github\\table-detection\\exps\\table-v2\\cluster.png', dpi=72, format='png')

"""
box_lists = read_box(maindir)
print(box_lists.shape)
for n in range(1, 16):
    clustering(box_lists, n_centroid=n, n_iters=50)
"""
plot_curve()
# -*- encoding: utf-8 -*-
# author: ronniecao
import os
import re
import matplotlib.pyplot as plt
import numpy


def load_log(path):
	with open(path, 'r') as fo:
		train_loss_iters, train_eval_iters, valid_eval_iters = [], [], []
		losses, class_losses, coord_losses, object_losses, nobject_losses, speeds = \
			[], [], [], [], [], []
		train_ious, train_objects, train_nobjects, train_recalls = \
			[], [], [], []
		valid_ious, valid_objects, valid_nobjects, valid_recalls = \
			[], [], [], []
			
		for line in fo:
			line = line.strip()
			
			# pattern1用于识别训练loss
			pattern1 = re.compile(r'([\d]+): ([\d\.]+), ([\d\.]+) avg')
			res1 = pattern1.findall(line)
			
			# pattern2用于识别训练evaluation
			pattern2 = re.compile(r'Region Avg IOU: ([\d\.]+), '
				r'Class: ([\d\.]+), Obj: ([\d\.]+), No Obj: ([\d\.]+), Avg Recall: ([\d\.]+)')
			res2 = pattern2.findall(line)

			if res1:
				losses.append(float(res1[0][1]))
			elif res2:
				train_ious.append(float(res2[0][0]))
				train_objects.append(float(res2[0][2]))
				train_nobjects.append(float(res2[0][3]))
				train_recalls.append(float(res2[0][4]))
				
	infos_dict = {
		'train_loss':{
			'iter': range(len(coord_losses)),
			'loss': losses}, 
		'train_eval':{
			'iter': range(len(train_ious)),
			'iou': train_ious, 'object': train_objects,
			'nobject': train_nobjects, 'recall': train_recalls}
	}
	
	return infos_dict

def curve_smooth(infos_dict, batch_size=1):
	new_infos_dict = {'train_loss':{}, 'train_eval': {}}

	k = [['train_loss', 'iter'], ['train_loss', 'loss'], ['train_eval', 'iter'], ['train_eval', 'iou'], 
		['train_eval', 'object']]
	for k1, k2 in k:
		bs = batch_size if k1 in ['train_loss', 'train_eval'] else 1
		new_list, data_list = [], infos_dict[k1][k2]
		for i in range(int(len(data_list) / bs)):
			batch = data_list[i*bs: (i+1)*bs]
			new_list.append(1.0 * sum(batch) / len(batch))
		new_infos_dict[k1][k2] = new_list

	return new_infos_dict

def plot_curve(infos_dict1, infos_dict2, infos_dict3):
	fig = plt.figure(figsize=(10, 5))

	plt.subplot(121)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['iou'], '.-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['iou'], '.-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['iou'], '.-', color='#FF6347')
	# plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('train iou value')
	plt.xlabel('# of iterations')
	plt.ylabel('iou')
	plt.xlim(xmin=0, xmax=150000)
	plt.ylim(ymin=0.5, ymax=1.0)

	plt.subplot(122)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['object'], '.-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['object'], '.-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['object'], '.-', color='#FF6347')
	# plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('train object value')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0, xmax=150000)
	plt.ylim(ymin=0.5, ymax=1.0)

	plt.show()
	# plt.savefig('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.png', dpi=120, format='png')


infos_dict1 = load_log('E:\\Github\\table-detection\\logs\\table-v6\\table-v1.txt')
infos_dict2 = load_log('E:\\Github\\table-detection\\logs\\table-v6\\table-v2.txt')
infos_dict3 = load_log('E:\\Github\\table-detection\\logs\\table-v6\\table-v4.txt')

infos_dict1 = curve_smooth(infos_dict1, batch_size=500)
infos_dict2 = curve_smooth(infos_dict2, batch_size=500)
infos_dict3 = curve_smooth(infos_dict3, batch_size=500)

plot_curve(infos_dict1, infos_dict2, infos_dict3)
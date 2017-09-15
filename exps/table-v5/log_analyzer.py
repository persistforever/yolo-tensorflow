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
			pattern1 = re.compile(r'\{TRAIN\} iter\[([\d]+)\], train_loss: ([\d\.]+), '
				r'coord_loss: ([\d\.]+), object_loss: ([\d\.]+), '
				r'nobject_loss: ([\d\.]+), image_nums: ([\d\.]+), speed: ([\d\.]+) images/s')
			res1 = pattern1.findall(line)
			
			# pattern2用于识别训练evaluation
			pattern2 = re.compile(r'\{TRAIN\} iter\[([\d]+)\], iou: ([\d\.]+), '
				r'object: ([\d\.]+), anyobject: ([\d\.]+), recall: ([\d\.]+)')
			res2 = pattern2.findall(line)
			
			# pattern3用于识别验证evaluation
			pattern3 = re.compile(r'\{VALID\} iter\[([\d]+)\], valid: iou: ([\d\.]+), '
				r'object: ([\d\.]+), nobject: ([\d\.]+), recall: ([\d\.]+)')
			res3 = pattern3.findall(line)

			if res1:
				train_loss_iters.append(int(res1[0][0]))
				losses.append(float(res1[0][1]))
				coord_losses.append(float(res1[0][2]))
				object_losses.append(float(res1[0][3]))
				nobject_losses.append(float(res1[0][4]))
				speeds.append(float(res1[0][6]))
			elif res2:
				train_eval_iters.append(int(res2[0][0]))
				train_ious.append(float(res2[0][1]))
				train_objects.append(float(res2[0][2]))
				train_nobjects.append(float(res2[0][3]))
				train_recalls.append(float(res2[0][4]))
			elif res3:
				valid_eval_iters.append(int(res3[0][0]))
				valid_ious.append(float(res3[0][1]))
				valid_objects.append(float(res3[0][2]))
				valid_nobjects.append(float(res3[0][3]))
				valid_recalls.append(float(res3[0][4]))
				
	infos_dict = {
		'train_loss':{
			'iter': train_loss_iters,
			'loss': losses, 'class_loss': class_losses,
			'coord_loss': coord_losses, 'object_loss': object_losses,
			'nobject_loss': nobject_losses, 'speed': speeds},
		'train_eval':{
			'iter': train_eval_iters,
			'iou': train_ious, 'object': train_objects,
			'nobject': train_nobjects, 'recall': train_recalls},
		'valid':{
			'iter': valid_eval_iters,
			'iou': valid_ious, 'object': valid_objects,
			'nobject': valid_nobjects, 'recall': valid_recalls}}
	
	return infos_dict

def curve_smooth(infos_dict, batch_size=1):
	new_infos_dict = {'train_loss':{}, 'train_eval': {}, 'valid': {}}

	k = [['train_loss', 'iter'], ['train_loss', 'loss'], ['train_eval', 'iter'], ['train_eval', 'iou'], 
		['train_eval', 'object'], ['valid', 'iter'], ['valid', 'iou'], ['valid', 'object']]
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

	plt.subplot(221)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['iou'], '.-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['iou'], '.-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['iou'], '.-', color='#FF6347')
	plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('train iou value')
	plt.xlabel('# of iterations')
	plt.ylabel('iou')
	plt.xlim(xmin=0, xmax=10000)
	plt.ylim(ymin=0.0, ymax=0.85)

	plt.subplot(222)
	p1 = plt.plot(infos_dict1['valid']['iter'], infos_dict1['valid']['iou'], 'o-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['valid']['iter'], infos_dict2['valid']['iou'], 'o-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['valid']['iter'], infos_dict3['valid']['iou'], 'o-', color='#FF6347')
	plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('valid iou value')
	plt.xlabel('# of iterations')
	plt.ylabel('iou')
	plt.xlim(xmin=0, xmax=10000)
	plt.ylim(ymin=0.0, ymax=0.85)

	plt.subplot(223)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['object'], '.-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['object'], '.-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['object'], '.-', color='#FF6347')
	plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('train object value')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0, xmax=10000)
	plt.ylim(ymin=0.0, ymax=1.0)

	plt.subplot(224)
	p1 = plt.plot(infos_dict1['valid']['iter'], infos_dict1['valid']['object'], 'o-', color='#66CDAA')
	p2 = plt.plot(infos_dict2['valid']['iter'], infos_dict2['valid']['object'], 'o-', color='#1E90FF')
	p3 = plt.plot(infos_dict3['valid']['iter'], infos_dict3['valid']['object'], 'o-', color='#FF6347')
	plt.legend((p1[0], p2[0], p3[0]), ('image + text', 'only image', 'image + all text'))
	plt.grid(True)
	plt.title('valid object value')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0, xmax=10000)
	plt.ylim(ymin=0.0, ymax=1.0)

	plt.show()
	# plt.savefig('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.png', dpi=120, format='png')


infos_dict1 = load_log('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.txt')
infos_dict2 = load_log('E:\\Github\\table-detection\\exps\\table-v4\\table-v6.txt')
infos_dict3 = load_log('E:\\Github\\table-detection\\exps\\table-v5\\table-v7.txt')

infos_dict1 = curve_smooth(infos_dict1, batch_size=10)
infos_dict2 = curve_smooth(infos_dict2, batch_size=10)
infos_dict3 = curve_smooth(infos_dict3, batch_size=10)

plot_curve(infos_dict1, infos_dict2, infos_dict3)
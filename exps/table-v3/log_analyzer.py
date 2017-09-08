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
		['valid', 'iter'], ['valid', 'iou']]
	for k1, k2 in k:
		new_list, data_list = [], infos_dict[k1][k2]
		for i in range(int(len(data_list) / batch_size)):
			batch = data_list[i*batch_size: (i+1)*batch_size]
			new_list.append(1.0 * sum(batch) / len(batch))
		new_infos_dict[k1][k2] = new_list

	return new_infos_dict

def plot_curve(infos_dict1):
	fig = plt.figure(figsize=(10, 5))

	plt.subplot(121)
	p1 = plt.plot(infos_dict1['train_loss']['iter'], infos_dict1['train_loss']['loss'], '.--', color='#66CDAA')
	# p2 = plt.plot(infos_dict2['train_loss']['iter'], infos_dict2['train_loss']['loss'], '.--', color='#1E90FF')
	# p3 = plt.plot(infos_dict3['train_loss']['iter'], infos_dict3['train_loss']['loss'], '.--', color='#FF6347')
	# p1 = plt.plot(infos_dict4['train_loss']['iter'], infos_dict4['train_loss']['loss'], '.--', color='#FFD700')
	# plt.legend((p1[0], p2[0], p3[0]), ('yolo-v1', 'data augmentation', 'change learning rate'))
	plt.grid(True)
	plt.title('train loss curve')
	plt.xlabel('# of iterations')
	plt.ylabel('loss')
	plt.xlim(xmin=0)
	# plt.ylim(ymin=0, ymax=5)

	plt.subplot(122)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['iou'], '.--', color='#66CDAA')
	# p2 = plt.plot(infos_dict2['valid']['iter'], infos_dict2['valid']['iou'], '.--', color='#1E90FF')
	# p3 = plt.plot(infos_dict3['valid']['iter'], infos_dict3['valid']['iou'], '.--', color='#FF6347')
	# p1 = plt.plot(infos_dict4['train_eval']['iter'], infos_dict4['train_eval']['iou'], '.--', color='#FFD700')
	# plt.legend((p1[0], p2[0], p3[0]), ('yolo-v1', 'data augmentation', 'change learning rate'))
	plt.grid(True)
	plt.title('train iou curve')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0)

	# plt.show()
	plt.savefig('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.png', dpi=72, format='png')


infos_dict1 = load_log('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.txt')

batch_size = 10
infos_dict1 = curve_smooth(infos_dict1, batch_size=batch_size)

plot_curve(infos_dict1)
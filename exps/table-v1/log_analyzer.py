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
			pattern1 = re.compile(r'\{TRAIN\} iter\[([\d]+)\], train loss: ([\d\.]+), '
				r'class_loss: ([\d\.]+), coord_loss: ([\d\.]+), object_loss: ([\d\.]+), '
				r'nobject_loss: ([\d\.]+), image_nums: ([\d\.]+), speed: ([\d\.]+) images/s')
			res1 = pattern1.findall(line)
			
			# pattern2用于识别训练evaluation
			pattern2 = re.compile(r'\{TRAIN\} iter\[([\d]+)\], iou: ([\d\.]+), '
				r'object: ([\d\.]+), nobject: ([\d\.]+), recall: ([\d\.]+)')
			res2 = pattern2.findall(line)
			
			# pattern3用于识别验证evaluation
			pattern3 = re.compile(r'\{VALID\} iter\[([\d]+)\], valid: iou: ([\d\.]+), '
				r'object: ([\d\.]+), nobject: ([\d\.]+), recall: ([\d\.]+)')
			res3 = pattern3.findall(line)

			if res1:
				train_loss_iters.append(int(res1[0][0]))
				losses.append(float(res1[0][1]))
				class_losses.append(float(res1[0][2]))
				coord_losses.append(float(res1[0][3]))
				object_losses.append(float(res1[0][4]))
				nobject_losses.append(float(res1[0][5]))
				speeds.append(float(res1[0][7]))
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

	new_list, data_list = [], infos_dict['train_loss']['iter']
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_list.append(1.0 * sum(batch) / len(batch))
	new_infos_dict['train_loss']['iter'] = new_list
	
	new_list, data_list = [], infos_dict['train_loss']['loss']
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_list.append(1.0 * sum(batch) / len(batch))
	new_infos_dict['train_loss']['loss'] = new_list
	
	new_list, data_list = [], infos_dict['valid']['iter']
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_list.append(1.0 * sum(batch) / len(batch))
	new_infos_dict['valid']['iter'] = new_list

	new_list, data_list = [], infos_dict['valid']['iou']
	for i in range(int(len(data_list) / batch_size)):
		batch = data_list[i*batch_size: (i+1)*batch_size]
		new_list.append(1.0 * sum(batch) / len(batch))
	new_infos_dict['valid']['iou'] = new_list

	return new_infos_dict
	

def plot_curve(infos_dict):
	fig = plt.figure(figsize=(10, 5))

	plt.subplot(121)
	p1 = plt.plot(infos_dict['train_loss']['iter'], infos_dict['train_loss']['loss'], '.--', color='#66CDAA')
	plt.grid(True)
	plt.title('train loss curve')
	plt.xlabel('# of iterations')
	plt.ylabel('loss')
	plt.xlim(xmin=0)
	plt.ylim(ymin=0, ymax=5)

	plt.subplot(122)
	p1 = plt.plot(infos_dict['valid']['iter'], infos_dict['valid']['iou'], '.--', color='#66CDAA')
	# plt.legend((p2[0], p3[0]), ('train_precision', 'valid_precision'))
	plt.grid(True)
	plt.title('valid iou curve')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0)

	plt.show()
 	# plt.savefig('E:\\Github\\table-detection\\exps\\table-v1\\table-v1.png', dpi=72, format='png')


infos_dict = load_log('E:\\Github\\table-detection\\exps\\table-v1\\table-v1.txt')
infos_dict = curve_smooth(infos_dict, batch_size=10)
plot_curve(infos_dict)
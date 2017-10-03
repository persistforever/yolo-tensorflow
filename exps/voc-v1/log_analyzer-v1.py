# -*- encoding: utf-8 -*-
# author: ronniecao
import os
import re
import matplotlib.pyplot as plt
import numpy


def load_log(path):
	with open(path, 'r') as fo:
		train_loss_iters = []
		losses, class_losses, coord_losses, object_losses, nobject_losses, class_losses = \
			[], [], [], [], [], []
		train_ious, train_objects, train_nobjects, train_recalls, train_class = \
			[], [], [], [], []
			
		for line in fo:
			line = line.strip()
			
			# pattern1用于识别训练loss
			pattern1 = re.compile(r'\{TRAIN\} \[([\d]+)\], train_loss: ([\d\.]+), coord_loss: ([\d\.]+), '
				r'object_loss: ([\d\.]+), nobject_loss: ([\d\.]+), class_loss: ([\d\.]+)')
			res1 = pattern1.findall(line)
			
			# pattern2用于识别训练evaluation
			pattern2 = re.compile(r'\{TRAIN\} \[([\d]+)\], IOU: ([\d\.]+), Object: ([\d\.]+), Noobject: ([\d\.]+), '
				r'Recall: ([\d\.]+), Class: ([\d\.]+)')
			res2 = pattern2.findall(line)

			if res1:
				losses.append(float(res1[0][1]))
			elif res2:
				train_ious.append(float(res2[0][1]))
				train_objects.append(float(res2[0][2]))
				train_nobjects.append(float(res2[0][3]))
				train_recalls.append(float(res2[0][4]))
				train_class.append(float(res2[0][5]))
				
	infos_dict = {
		'train_loss':{
			'iter': range(len(coord_losses)),
			'loss': losses}, 
		'train_eval':{
			'iter': range(len(train_ious)),
			'iou': train_ious, 
			'object': train_objects,
			'nobject': train_nobjects, 
			'recall': train_recalls,
			'class': train_class}
	}
	
	return infos_dict

def curve_smooth(infos_dict, batch_size=1):
	new_infos_dict = {'train_loss':{}, 'train_eval': {}}

	k = [['train_eval', 'iter'], ['train_eval', 'iou'], ['train_eval', 'object'], ['train_eval', 'class']]
	for k1, k2 in k:
		bs = batch_size if k1 in ['train_loss', 'train_eval'] else 1
		new_list, data_list = [], infos_dict[k1][k2]
		for i in range(int(len(data_list) / bs)):
			batch = data_list[i*bs: (i+1)*bs]
			new_list.append(1.0 * sum(batch) / len(batch))
		new_infos_dict[k1][k2] = new_list

	return new_infos_dict

def plot_curve(infos_dict1):
	fig = plt.figure(figsize=(10, 5))

	plt.subplot(131)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['iou'], '.-', color='#66CDAA')
	# p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['iou'], '.-', color='#1E90FF')
	# p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['iou'], '.-', color='#FF6347')
	# p4 = plt.plot(infos_dict4['train_eval']['iter'], infos_dict4['train_eval']['iou'], '.-', color='#FFB90F')
	# p5 = plt.plot(infos_dict5['train_eval']['iter'], infos_dict5['train_eval']['iou'], '.-', color='#8B658B')
	# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('image + part text', 'only image', 'image + whole text1', 'image + whole text2'))
	plt.grid(True)
	plt.title('train iou value')
	plt.xlabel('# of iterations')
	plt.ylabel('iou')
	plt.xlim(xmin=0, xmax=30000)
	# plt.ylim(ymin=0.85, ymax=0.92)

	plt.subplot(132)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['object'], '.-', color='#66CDAA')
	# p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['object'], '.-', color='#1E90FF')
	# p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['object'], '.-', color='#FF6347')
	# p4 = plt.plot(infos_dict4['train_eval']['iter'], infos_dict4['train_eval']['object'], '.-', color='#FFB90F')
	# p5 = plt.plot(infos_dict5['train_eval']['iter'], infos_dict5['train_eval']['object'], '.-', color='#8B658B')
	# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('image + part text', 'only image', 'image + whole text1', 'image + whole text2'))
	plt.grid(True)
	plt.title('train object value')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0, xmax=30000)
	# plt.ylim(ymin=0.95, ymax=0.99)

	plt.subplot(133)
	p1 = plt.plot(infos_dict1['train_eval']['iter'], infos_dict1['train_eval']['class'], '.-', color='#66CDAA')
	# p2 = plt.plot(infos_dict2['train_eval']['iter'], infos_dict2['train_eval']['object'], '.-', color='#1E90FF')
	# p3 = plt.plot(infos_dict3['train_eval']['iter'], infos_dict3['train_eval']['object'], '.-', color='#FF6347')
	# p4 = plt.plot(infos_dict4['train_eval']['iter'], infos_dict4['train_eval']['object'], '.-', color='#FFB90F')
	# p5 = plt.plot(infos_dict5['train_eval']['iter'], infos_dict5['train_eval']['object'], '.-', color='#8B658B')
	# plt.legend((p1[0], p2[0], p3[0], p4[0]), ('image + part text', 'only image', 'image + whole text1', 'image + whole text2'))
	plt.grid(True)
	plt.title('train object value')
	plt.xlabel('# of iterations')
	plt.ylabel('accuracy')
	plt.xlim(xmin=0, xmax=30000)
	# plt.ylim(ymin=0.95, ymax=0.99)

	plt.show()
	# plt.savefig('E:\\Github\\table-detection\\exps\\table-v3\\table-v3.png', dpi=120, format='png')


infos_dict1 = load_log('E:\\Github\\yolo-tensorflow\\logs\\voc-v1\\train-v1.txt')

infos_dict1 = curve_smooth(infos_dict1, batch_size=100)

plot_curve(infos_dict1)
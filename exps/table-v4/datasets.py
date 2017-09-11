# -*- coding: utf-8 -*-
# author: ronniecao
import json
import os
import shutil
import random
import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import platform

colors = {
	'word': [150, 150, 150], # 灰色
	'date': [30, 144, 255], # 蓝色
	'digit': [67, 205, 128], # 绿色
	'line': [54, 54, 54], # 黑色
	'page': [159, 121, 238], # 紫色
	'table': [255, 99, 71], # 红色
	'picture': [255, 236, 139], # 黄色
}
for key in colors:
	colors[key] = [0, 0, 0]

def draw_image(contents_dict, maindir):
	n_processed = 1
	for docid in contents_dict:
		print('Draw Images: docid: %s, rate: %.2f%%' % (docid, 100.0 * n_processed / len(contents_dict)))
		if not os.path.exists(maindir):
			os.mkdir(maindir)
		if not os.path.exists(os.path.join(maindir, docid)):
			os.mkdir(os.path.join(maindir, docid))
		if not os.path.exists(os.path.join(maindir, docid, 'png_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_notable'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_line_table')):
			os.mkdir(os.path.join(maindir, docid, 'png_line_table'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_noline_table')):
			os.mkdir(os.path.join(maindir, docid, 'png_noline_table'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_line_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_line_notable'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_noline_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_noline_notable'))
		for pageid in contents_dict[docid]:
			shape = contents_dict[docid][pageid]['size']
			if len(contents_dict[docid][pageid]['tables']) == 0:
				# 如果没有表格，则画图并保存在no_table文件夹中 
				image = numpy.zeros((shape[1], shape[0], 3), dtype='uint8') + 255
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 100:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['picture']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 2:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['word']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 5:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['page']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 3:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['date']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 4:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['digit']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 1:
						image[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['line']
					if text['type'] not in [1, 2, 3, 4, 5, 100]:
						print(text)
				image_path = os.path.join(
					maindir, docid, 'png_notable', '%s_%s_notable.png' % (docid, pageid))
				misc.imsave(image_path, image)
			else:
				image_line = numpy.zeros((shape[1], shape[0], 3), dtype='uint8') + 255
				image_noline = numpy.zeros((shape[1], shape[0], 3), dtype='uint8') + 255
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 100:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['picture']
						image_noline[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['picture']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 2:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['word']
						image_noline[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['word']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 5:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['page']
						image_noline[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['page']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 3:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['date']
						image_noline[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['date']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 4:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['digit']
						image_noline[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['digit']
				for text in contents_dict[docid][pageid]['boxes']:
					if text['type'] == 1:
						image_line[text['position'][2]:text['position'][3],
							text['position'][0]:text['position'][1], :] = colors['line']
				image_line_path = os.path.join(
					maindir, docid, 'png_line_notable',
					'%s_%s_linenotable.png' % (docid, pageid))
				misc.imsave(image_line_path, image_line)
				image_noline_path = os.path.join(
					maindir, docid, 'png_noline_notable',
					'%s_%s_nolinenotable.png' % (docid, pageid))
				misc.imsave(image_noline_path, image_noline)
				for table in contents_dict[docid][pageid]['tables']:
					image_line[table['position'][2]:table['position'][3],
						table['position'][0]-1:table['position'][0]+1, :] = colors['table']
					image_line[table['position'][2]:table['position'][3],
						table['position'][1]-1:table['position'][1]+1, :] = colors['table']
					image_line[table['position'][2]-1:table['position'][2]+1,
						table['position'][0]:table['position'][1], :] = colors['table']
					image_line[table['position'][3]-1:table['position'][3]+1,
						table['position'][0]:table['position'][1], :] = colors['table']
					image_noline[table['position'][2]:table['position'][3],
						table['position'][0]-1:table['position'][0]+1, :] = colors['table']
					image_noline[table['position'][2]:table['position'][3],
						table['position'][1]-1:table['position'][1]+1, :] = colors['table']
					image_noline[table['position'][2]-1:table['position'][2]+1,
						table['position'][0]:table['position'][1], :] = colors['table']
					image_noline[table['position'][3]-1:table['position'][3]+1,
						table['position'][0]:table['position'][1], :] = colors['table']
				image_line_path = os.path.join(
					maindir, docid, 'png_line_table',
					'%s_%s_linetable.png' % (docid, pageid))
				misc.imsave(image_line_path, image_line)
				image_noline_path = os.path.join(
					maindir, docid, 'png_noline_table', 
					'%s_%s_nolinetable.png' % (docid, pageid))
				misc.imsave(image_noline_path, image_noline)
		n_processed += 1
		
def create_labels(contents_dict, maindir):
	n_processed = 1
	for docid in contents_dict:
		print('Write Labels: docid: %s, rate: %.2f%%' % (docid, 100.0 * n_processed / len(contents_dict)))
		if not os.path.exists(maindir):
			os.mkdir(maindir)
		if not os.path.exists(os.path.join(maindir, docid)):
			os.mkdir(os.path.join(maindir, docid))
		if not os.path.exists(os.path.join(maindir, docid, 'png_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_notable'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_line_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_line_notable'))
		if not os.path.exists(os.path.join(maindir, docid, 'png_noline_notable')):
			os.mkdir(os.path.join(maindir, docid, 'png_noline_notable'))
		for pageid in contents_dict[docid]:
			shape = contents_dict[docid][pageid]['size']
			if len(contents_dict[docid][pageid]['tables']) == 0:
				label_path = os.path.join(
					maindir, docid, 'png_notable', 
					'%s_%s_notable.txt' % (docid, pageid))
				with open(label_path, 'w') as fw:
					pass
			else:
				label_path = os.path.join(
					maindir, docid, 'png_noline_notable', 
					'%s_%s_nolinenotable.txt' % (docid, pageid))
				with open(label_path, 'w') as fw:
					for table in contents_dict[docid][pageid]['tables']:
						x = 1.0 * (table['position'][0] + table['position'][1]) / (2.0 * shape[0])
						y = 1.0 * (table['position'][2] + table['position'][3]) / (2.0 * shape[1])
						w = 1.0 * (table['position'][1] - table['position'][0]) / shape[0]
						h = 1.0 * (table['position'][3] - table['position'][2]) / shape[1]
						fw.writelines('0 %.8f %.8f %.8f %.8f\n' % (x, y, w, h))
				label_path = os.path.join(
					maindir, docid, 'png_line_notable', 
					'%s_%s_linenotable.txt' % (docid, pageid))
				with open(label_path, 'w') as fw:
					for table in contents_dict[docid][pageid]['tables']:
						x = 1.0 * (table['position'][0] + table['position'][1]) / (2.0 * shape[0])
						y = 1.0 * (table['position'][2] + table['position'][3]) / (2.0 * shape[1])
						w = 1.0 * (table['position'][1] - table['position'][0]) / shape[0]
						h = 1.0 * (table['position'][3] - table['position'][2]) / shape[1]
						fw.writelines('0 %.8f %.8f %.8f %.8f\n' % (x, y, w, h))
				
def create_datasets(contents_dict, maindir):
	positives, negatives = [], []
	train_list, valid_list, test_list = [], [], []
	n_processed = 1
	for docid in contents_dict:
		print('Create Dataset: docid: %s, rate: %.2f%%' % (docid, 100.0 * n_processed / len(contents_dict)))
		for pageid in contents_dict[docid]:
			if len(contents_dict[docid][pageid]['tables']) == 0:
				negatives.append('%s_%s_0' % (docid, pageid))
			else:
				positives.append('%s_%s_1' % (docid, pageid))
		n_processed += 1
	random.shuffle(positives)
	random.shuffle(negatives)
	train_list.extend(positives[0: int(len(positives) * 0.0)])
	train_list.extend(negatives[0: int(len(negatives) * 0.0)])
	valid_list.extend(positives[int(len(positives) * 0.0): int(len(positives) * 0.0)])
	valid_list.extend(negatives[int(len(negatives) * 0.0): int(len(negatives) * 0.0)])
	test_list.extend(positives[int(len(positives) * 0.0):])
	test_list.extend(negatives[int(len(negatives) * 0.0):])
	random.shuffle(train_list)
	random.shuffle(valid_list)
	random.shuffle(test_list)
	
	def _writes(data_list, fname):
		with open(os.path.join(maindir, fname), 'w') as fw:
			for filename in data_list:
				[docid, pageid, tag] = filename.split('_')
				if tag == '0':
					filename = os.path.join(
						maindir, 'JPEGImages', docid, 'png_notable', 
						'%s_%s_%s.png' % (docid, pageid, 'notable'))
					fw.writelines('%s\n' % (filename))
				else:
					filename = os.path.join(
						maindir, 'JPEGImages', docid, 'png_noline_notable', 
						'%s_%s_%s.png' % (docid, pageid, 'nolinenotable'))
					fw.writelines('%s\n' % (filename))
					filename = os.path.join(
						maindir, 'JPEGImages', docid, 'png_line_notable', 
						'%s_%s_%s.png' % (docid, pageid, 'linenotable'))
					fw.writelines('%s\n' % (filename))
					
	_writes(train_list, 'train.txt')
	_writes(valid_list, 'valid.txt')
	_writes(test_list, 'test.txt')

def read_json(path):
	with open(path, 'r') as fo:
		contents_dict = json.load(fo, encoding='utf8')
		return contents_dict

def write_json(contents_dict, path):
	with open(path, 'w') as fw:
		fw.write(json.dumps(contents_dict, indent=4))

def uint_test1(maindir):
	train_dict, valid_dict, test_dict = {}, {}, {}
	with open(os.path.join(maindir, 'train.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if filename not in train_dict:
				train_dict[filename] = None
	with open(os.path.join(maindir, 'valid.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if filename not in train_dict:
				valid_dict[filename] = None
	with open(os.path.join(maindir, 'test.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if filename not in train_dict:
				test_dict[filename] = None
	for filename in train_dict:
		if filename in valid_dict:
			print('训练集中的数据出现在验证集中！')
			return
		if filename in valid_dict:
			print('训练集中的数据出现在测试集中！')
			return
	for filename in valid_dict:
		if filename in test_dict:
			print('验证集中的数据出现在测试集中！')
			return
	print('单元测试：训练集、验证集、测试集是否有重合？\t无重合， 单元测试通过！')
	
def uint_test2(maindir):
	with open(os.path.join(maindir, 'train.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if not os.path.exists(filename):
				print('不存在此图片：%s' % (filename))
				return
	with open(os.path.join(maindir, 'valid.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if not os.path.exists(filename):
				print('不存在此图片：%s' % (filename))
				return
	with open(os.path.join(maindir, 'test.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			if not os.path.exists(filename):
				print('不存在此图片：%s' % (filename))
				return
	print('单元测试：训练集、验证集、测试集中的图片是否都存在？\t均存在， 单元测试通过！')
	
def uint_test3(maindir):
	with open(os.path.join(maindir, 'train.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			labelname = filename
			labelname.replace('JPEGImages', 'labels')
			labelname.replace('png', 'txt')
			if not os.path.exists(labelname):
				print('不存在此标签：%s' % (labelname))
				return
	with open(os.path.join(maindir, 'valid.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			labelname = filename
			labelname.replace('JPEGImages', 'labels')
			labelname.replace('png', 'txt')
			if not os.path.exists(labelname):
				print('不存在此标签：%s' % (labelname))
				return
	with open(os.path.join(maindir, 'test.txt'), 'r') as fo:
		for line in fo.readlines():
			filename = line.strip()
			labelname = filename
			labelname.replace('JPEGImages', 'labels')
			labelname.replace('png', 'txt')
			if not os.path.exists(labelname):
				print('不存在此标签：%s' % (labelname))
				return
	print('单元测试：图片对应的标签是否都存在？\t均存在， 单元测试通过！')


if 'Windows' in platform.platform():
	contents_dict = load_table_json('E:\\Temporal\Python\darknet-master\datasets\\table-png\JPEGImages')
	write_json(contents_dict, 'E:\\Temporal\Python\darknet-master\datasets\\table-png\\texts.json')
	contents_dict = read_json('E:\\Temporal\Python\darknet-master\datasets\\table-png\\texts.json')
	draw_image(contents_dict, 'E:\\Temporal\Python\darknet-master\datasets\\table-png\JPEGImages')
	create_labels(contents_dict, 'E:\\Temporal\Python\darknet-master\datasets\\table-png\labels')
	create_datasets(contents_dict, 'E:\\Temporal\Python\darknet-master\datasets\\table-png')
	uint_test1('E:\\Temporal\Python\darknet-master\datasets\\table-png')
	uint_test2('E:\\Temporal\Python\darknet-master\datasets\\table-png')
	uint_test3('E:\\Temporal\Python\darknet-master\datasets\\table-png')
elif 'Linux' in platform.platform():
	contents_dict = read_json('/home/caory/github/table-detection/data/table-v2/texts.json')
	draw_image(contents_dict, '/home/caory/github/table-detection/data/table-v2/JPEGImages')
	# create_labels(contents_dict, '/home/ronniecao/yolo/darknet/datasets/table-test/labels')
	# create_datasets(contents_dict, '/home/ronniecao/yolo/darknet/datasets/table-test')
	# uint_test1('/home/ronniecao/yolo/darknet/datasets/table-test')
	# uint_test2('/home/ronniecao/yolo/darknet/datasets/table-test')
	# uint_test3('/home/ronniecao/yolo/darknet/datasets/table-test')
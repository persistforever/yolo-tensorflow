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
import sys

colors = {
    'word': [150, 150, 150], # 灰色
    'date': [30, 144, 255], # 蓝色
    'digit': [67, 205, 128], # 绿色
    'line': [54, 54, 54], # 黑色
    'page': [159, 121, 238], # 紫色
    'table': [255, 99, 71], # 红色
    'picture': [255, 236, 139], # 黄色
}

def draw_image(contents_dict, maindir):
	n_processed = 1
	for docid in contents_dict:
		n_processed += 1
		print('Draw Images: docid: %s, rate: %.2f%%' % (docid, 100.0 * n_processed / len(contents_dict)))
		sys.stdout.flush()

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

				new_text = []
				for idx in range(len(contents_dict[docid][pageid]['boxes'])):
					text = contents_dict[docid][pageid]['boxes'][idx]
					if 0 <= text['position'][0] < shape[0] and 0 <= text['position'][1] < shape[0] and \
						0 <= text['position'][2] < shape[1] and 0 <= text['position'][3] < shape[1]:
						new_text.append(text)
				contents_dict[docid][pageid]['boxes'] = new_text

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
		
def create_labels(contents_dict, maindir):
    dataset = []
    
    n_processed = 0
    for docid in contents_dict:
        n_processed += 1
        print('Write Labels: docid: %s, rate: %.2f%%' % (docid, 100.0 * n_processed / len(contents_dict)))
        sys.stdout.flush()

        if not os.path.exists(maindir):
            os.mkdir(maindir)
        if not os.path.exists(os.path.join(maindir, 'JPEGImages', docid)):
            os.mkdir(os.path.join(maindir, 'JPEGImages', docid))
        if not os.path.exists(os.path.join(maindir, 'JPEGImages', docid, 'png_notable')):
            os.mkdir(os.path.join(maindir, 'JPEGImages', docid, 'png_notable'))
        if not os.path.exists(os.path.join(maindir, 'JPEGImages', docid, 'png_line_notable')):
            os.mkdir(os.path.join(maindir, 'JPEGImages', docid, 'png_line_notable'))
        if not os.path.exists(os.path.join(maindir, 'JPEGImages', docid, 'png_noline_notable')):
            os.mkdir(os.path.join(maindir, 'JPEGImages', docid, 'png_noline_notable'))
        
        for pageid in contents_dict[docid]:
            shape = contents_dict[docid][pageid]['size']
            if len(contents_dict[docid][pageid]['tables']) == 0:
                picpath = os.path.join(maindir, 'JPEGImages', docid, 'png_notable', \
                    '%s_%s_notable.txt' % (docid, pageid))
                label = [picpath]
                if os.path.exists(label[0]):
                    dataset.append(' '.join(label))
            else:
                # noline
                picpath = os.path.join(maindir, 'JPEGImages', docid, 'png_noline_notable', \
                    '%s_%s_nolinenotable.txt' % (docid, pageid))
                label = [picpath]
                for table in contents_dict[docid][pageid]['tables']:
                    left = str(int(table['position'][0]))
                    right = str(int(table['position'][1]))
                    top = str(int(table['position'][2]))
                    bottom = str(int(table['position'][3]))
                    label.extend([left, right, top, bottom, '1'])
                    if os.path.exists(label[0]):
                        dataset.append(' '.join(label))

                # line
                picpath = os.path.join(maindir, 'JPEGImages', docid, 'png_line_notable', \
                    '%s_%s_linenotable.txt' % (docid, pageid))
                label = [picpath]
                for table in contents_dict[docid][pageid]['tables']:
                    left = str(int(table['position'][0]))
                    right = str(int(table['position'][1]))
                    top = str(int(table['position'][2]))
                    bottom = str(int(table['position'][3]))
                    label.extend([left, right, top, bottom, '1'])
                    if os.path.exists(label[0]):
                        dataset.append(' '.join(label))

    random.shuffle(dataset)

    train_list, valid_list, test_list = [], [], []
    train_list.extend(dataset[0: int(len(dataset) * 0.9)])
    valid_list.extend(dataset[int(len(dataset) * 0.9): int(len(dataset) * 0.95)])
    test_list.extend(dataset[int(len(dataset) * 0.95):])
	
    def _writes(data_list, fname):
        with open(os.path.join(maindir, fname), 'w') as fw:
            for label in data_list:
                fw.writelines('%s\n' % (label))
					
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
    # create_labels(contents_dict, '/home/caory/github/table-detection/data/table-v2/')

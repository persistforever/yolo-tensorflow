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

def load_source(maindir):
	n_processed = 0
    for docid in os.listdir(maindir)[0:2]:
		n_processed += 1
		logging.info('Load Source: doc: %s, rate: %.2f%%' % (
			docid, 100.0 * (idx+1) / len(dirlist)))

        json_path = os.path.join(maindir, docid, 'pages_with_tables')
        data = read_json(json_path)
        for pageid in data:
            print(data[pageid].keys())
            size = data[pageid]['size']
            texts, curves, others, tables = [], [], [], []
			# 获取表格框
			pad = 2
			for box in data[pageid]['tables']:
				pos = [int(math.floor(float(box[0])) - pad), \
					int(math.ceil(float(box[2])) + pad), \
					int(math.floor(float(size[1]-box[3])) - pad), \
					int(math.ceil(float(size[1]-box[1])) + pad)]
				tables.append({'position': pos, 'lines': [], 'texts': [], 'cells': []})
			# 获取文本框
			for text in data[pageid]['texts']:
				# 获取每一个字符的位置
				chars = []
				for char in text['chars']:
					pos = [int(math.floor(float(char['box'][0]))),
						int(math.floor(float(char['box'][2]))),
						int(math.floor(float(size[1]-char['box'][3]))),
						int(math.floor(float(size[1]-char['box'][1])))]
					chars.append({'position': pos, 'sentence': char['text'].strip()})
				# 对于距离近的字符进行合并
				for char in chars:
					merged = False
					for i in range(len(texts)):
						box = texts[i]
						if char['position'][2] == texts[i]['position'][2] and \
							char['position'][3] == texts[i]['position'][3] and \
							text['type'] == texts[i]['type']:
							if abs(char['position'][0] - texts[i]['position'][1]) <= 5:
								texts[i]['position'][1] = char['position'][1]
								merged = True
								break
							elif abs(char['position'][1] - texts[i]['position'][0]) <= 5:
								texts[i]['position'][0] = char['position'][0]
								merged = True
								break
					if not merged:
						texts.append({'position': char['position'], 'type': text['type'],
									'sentence': text['text'].strip()})
			new_texts = []
			for text in texts:
				for table in tables:
					if text['position'][0] >= table['position'][0] and \
						text['position'][1] <= table['position'][1] and \
						text['position'][2] >= table['position'][2] and \
						text['position'][3] <= table['position'][3]:
						table['texts'].append(text)
						break
				else:
					new_texts.append(text)
			texts = new_texts
			# 对于页码进行特殊识别
			left_bottom, middle_bottom, right_bottom = [], [], []
			for i in range(len(texts)):
				xrate = float((texts[i]['position'][0]+texts[i]['position'][1]) / 2) / size[0]
				yrate = float((texts[i]['position'][2]+texts[i]['position'][3]) / 2) / size[1]
				if 0.02 <= xrate <= 0.1 and 0.85 <= yrate <= 1.0 and \
					texts[i]['type'] == 4:
					left_bottom.append(i)
				elif 0.45 <= xrate <= 0.55 and 0.85 <= yrate <= 1.0 and \
					texts[i]['type'] == 4:
					middle_bottom.append(i)
				elif 0.90 <= xrate <= 0.94 and 0.85 <= yrate <= 1.0 and \
					texts[i]['type'] == 4:
					right_bottom.append(i)
			if len(left_bottom) != 0:
				i = max(left_bottom, key=lambda x: texts[x]['position'][3])
				texts[i]['type'] = 5
			elif len(right_bottom) != 0:
				i = max(right_bottom, key=lambda x: texts[x]['position'][3])
				texts[i]['type'] = 5
			elif len(middle_bottom) != 0:
				i = max(middle_bottom, key=lambda x: texts[x]['position'][3])
				texts[i]['type'] = 5
			# 将下划线文本框改为表格框
			new_texts = []
			for text in texts:
				isline = True
				if 'sentence' in text and text['type'] == 2:
					for s in text['sentence']:
						if s != '_':
							isline = False
					if isline and len(text['sentence']) >= 3:
						pos = [text['position'][0], text['position'][1], 
							text['position'][3]-1, text['position'][3]]
						curves.append({'position': pos, 'type': 1})
					else:
						new_texts.append(text)
				else:
					new_texts.append(text)
			texts = new_texts
			# 获取其他框（图片等）
			for other in data[pageid]['others']:
				pos = [int(math.floor(float(other['box'][0]))), \
					int(math.floor(float(other['box'][2]))), \
					int(math.floor(float(size[1]-other['box'][3]))), \
					int(math.floor(float(size[1]-other['box'][1])))]
				others.append({'position': pos, 'type': other['type']})
			# 获取每一个线条的位置
			curves = []
			curve_width = 2
			for curve in data[pageid]['curves']:
				pos = [int(math.floor(float(curve['box'][0]))), \
					int(math.floor(float(curve['box'][2]))), \
					int(math.floor(float(size[1]-curve['box'][3]))), \
					int(math.floor(float(size[1]-curve['box'][1])))]
				if pos[1] - pos[0] <= curve_width and pos[3] - pos[2] > curve_width:
					pos[1] = pos[0]
					line = {'position': pos, 'type': curve['type']}
				elif pos[1] - pos[0] > curve_width and pos[3] - pos[2] <= curve_width:
					pos[3] = pos[2]
					line = {'position': pos, 'type': curve['type']}
				for table in tables:
					if line['position'][0] >= table['position'][0] and \
						line['position'][1] <= table['position'][1] and \
						line['position'][2] >= table['position'][2] and \
						line['position'][3] <= table['position'][3] and \
						line['type'] == 1:
						table['lines'].append(line)
						break
				else:
					curves.append(line)


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
    contents_dict = load_source('/home/wangxu/data/pdf2jpg_v4/output')
    # draw_image(contents_dict, '/home/caory/github/table-detection/data/table-v2/JPEGImages')
    # create_labels(contents_dict, '/home/caory/github/table-detection/data/table-v2/')

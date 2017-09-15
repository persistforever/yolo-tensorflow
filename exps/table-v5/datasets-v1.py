# -*- coding: utf-8 -*-
# author: ronniecao
import json
import os
import shutil
import codecs
import random
import math
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy import misc
import platform
import sys
from pyltp import Segmentor

colors = {
    'word': [150, 150, 150], # 灰色
    'date': [30, 144, 255], # 蓝色
    'digit': [67, 205, 128], # 绿色
    'line': [54, 54, 54], # 黑色
    'page': [159, 121, 238], # 紫色
    'table': [255, 99, 71], # 红色
    'picture': [255, 236, 139], # 黄色
}

def read_word_vector(path):
    word_dict, word_vector, n = {}, [], 0

    with codecs.open(path, 'r', 'utf8') as fo:
        for line in fo:
            [word, vector] = line.strip().split('\t')
            vector = [float(t) for t in vector.split(' ')]
            word_dict[n] = word
            n += 1
            word_vector.append(vector)

    for j in range(3):
        col_min = min([word_vector[i][j] for i in range(n)])
        col_max = max([word_vector[i][j] for i in range(n)])
        for i in range(n):
            word_vector[i][j] = 1.0 * (word_vector[i][j] - col_min) / (col_max - col_min)

    new_word_dict = {}
    for i in range(n):
        new_word_dict[word_dict[i]] = word_vector[i]

    return new_word_dict

def load_source(maindir, word_dict):
    n_processed = 0
    contents_dict = {}
    segmentor = Segmentor()
    segmentor.load('/home/caory/github/table-detection/data/table-v5/ltp_data/cws.model')

    dirlist = os.listdir(maindir)
    for docid in dirlist:
        n_processed += 1
        print('Load Source: doc: %s, rate: %.2f%%' % (
            docid, 100.0 * n_processed / len(dirlist)))
        sys.stdout.flush()
        contents_dict[docid] = {}

        json_path = os.path.join(maindir, docid, 'pages_with_tables')
        if not os.path.exists(json_path):
            continue

        data = read_json(json_path)
        for pageid in data:
            contents_dict[docid][pageid] = {}
            size = data[pageid]['size']
            texts, curves, others, tables = [], [], [], []
			
            # 获取表格框
            pad, offset = 2, 5
            for box in data[pageid]['tables']:
                left = max(offset, int(math.floor(float(box[0])) - pad))
                right = min(int(math.ceil(float(box[2])) + pad), size[0]-offset)
                top = max(offset, int(math.floor(float(size[1]-box[3])) - pad))
                bottom = min(int(math.ceil(float(size[1]-box[1])) + pad), size[1]-offset)
                if 0 <= left <= right < size[0] and 0 <= top <= bottom < size[1]:
                    tables.append({'position': [left, right, top, bottom]})
			
            # 获取文本框
            for text in data[pageid]['texts']:
                # 获取每一个字符的位置
                chars = []
                for char in text['chars']:
                    left = int(math.floor(float(char['box'][0])))
                    right = int(math.floor(float(char['box'][2])))
                    top = int(math.floor(float(size[1]-char['box'][3])))
                    bottom = int(math.floor(float(size[1]-char['box'][1])))
                    if 0 <= left <= right < size[0] and 0 <= top <= bottom < size[1]:
                        chars.append({'position': [left, right, top, bottom], 'sentence': char['text'].strip()})
                
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
			
            # 对于页码进行特殊识别
            for i in range(len(texts)):
                top = texts[i]['position'][2]
                bottom = texts[i]['position'][3]
                if 1.0 * top / size[1] <= 0.85:
                    continue
                is_page = True

                for j in range(len(texts)):
                    if j == i:
                        continue
                    other_top = texts[j]['position'][2]
                    other_bottom = texts[j]['position'][3]
                    if other_bottom >= top:
                        is_page = False
                        break
                
                if is_page:
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
                left = int(math.floor(float(other['box'][0])))
                right = int(math.floor(float(other['box'][2])))
                top = int(math.floor(float(size[1]-other['box'][3])))
                bottom = int(math.floor(float(size[1]-other['box'][1])))
                if 0 <= left <= right < size[0] and 0 <= top <= bottom < size[1]:
                    others.append({'position': [left, right, top, bottom], 'type': other['type']})
			
            # 获取每一个线条的位置
            curves = []
            curve_width = 2
            for curve in data[pageid]['curves']:
                left = int(math.floor(float(curve['box'][0])))
                right = int(math.floor(float(curve['box'][2])))
                top = int(math.floor(float(size[1]-curve['box'][3])))
                bottom = int(math.floor(float(size[1]-curve['box'][1])))
                if right - left <= curve_width and bottom - top > curve_width:
                    right = left
                    line = {'position': [left, right, top, bottom], 'type': curve['type']}
                elif right - left > curve_width and bottom - top <= curve_width:
                    bottom = top
                    line = {'position': [left, right, top, bottom], 'type': curve['type']}
                if line:
                    if 0 <= line['position'][0] <= line['position'][1] < size[0] and \
                        0 <= line['position'][2] <= line['position'][3] < size[1]:
                        curves.append(line)
            
            contents_dict[docid][pageid] = {
                'texts': texts, 'size': size, 'tables': tables,
                'others': others, 'curves': curves}

    return contents_dict

def draw_image(contents_dict, maindir):
    n_processed = 0
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

                for text in contents_dict[docid][pageid]['others']:
                    if text['type'] == 100:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['picture']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 2:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['word']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 5:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['page']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 3:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['date']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 4:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['digit']

                for text in contents_dict[docid][pageid]['curves']:
                    if text['type'] == 1:
                        image[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['line']

                image_path = os.path.join(
                    maindir, docid, 'png_notable', '%s_%s_notable.png' % (docid, pageid))
                misc.imsave(image_path, image)
            else:
                image_line = numpy.zeros((shape[1], shape[0], 3), dtype='uint8') + 255
                image_noline = numpy.zeros((shape[1], shape[0], 3), dtype='uint8') + 255

                for text in contents_dict[docid][pageid]['others']:
                    if text['type'] == 100:
                        image_line[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['picture']
                        image_noline[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['picture']
                            
                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 2:
                        image_line[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['word']
                        image_noline[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['word']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 5:
                        image_line[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['page']
                        image_noline[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['page']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 3:
                        image_line[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['date']
                        image_noline[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['date']

                for text in contents_dict[docid][pageid]['texts']:
                    if text['type'] == 4:
                        image_line[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['digit']
                        image_noline[text['position'][2]:text['position'][3],
                            text['position'][0]:text['position'][1], :] = colors['digit']

                for text in contents_dict[docid][pageid]['curves']:
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
                    '%s_%s_notable.png' % (docid, pageid))
                label = [picpath]
                if os.path.exists(label[0]):
                    dataset.append(' '.join(label))
            else:
                # noline
                picpath = os.path.join(maindir, 'JPEGImages', docid, 'png_noline_notable', \
                    '%s_%s_nolinenotable.png' % (docid, pageid))
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
                    '%s_%s_linenotable.png' % (docid, pageid))
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
    # word_dict = read_word_vector('/home/caory/github/table-detection/data/table-v5/word_vector_3.txt')
    # contents_dict = load_source('/home/wangxu/data/pdf2jpg_v4/output', word_dict)
    # write_json(contents_dict, '/home/caory/github/table-detection/data/table-v5/texts.json')
    contents_dict = read_json('/home/caory/github/table-detection/data/table-v5/texts.json')
    draw_image(contents_dict, '/home/caory/github/table-detection/data/table-v3/JPEGImages')
    # create_labels(contents_dict, '/home/caory/github/table-detection/data/table-v2/')

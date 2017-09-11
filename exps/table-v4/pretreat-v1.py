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
import logging

if 'Windows' in platform.platform():
	logging.basicConfig(
		level=logging.INFO,
		filename='E:\\Temporal\Python\darknet-master\log\pretreat.txt',
		filemode='w')
elif 'Linux' in platform.platform():
	logging.basicConfig(
		level=logging.INFO,
		filename='/home/ronniecao/yolo/darknet/log/pretreat.txt',
		filemode='a+')

colors = {
	'word': [150, 150, 150], # 灰色
	'date': [30, 144, 255], # 蓝色
	'digit': [67, 205, 128], # 绿色
	'line': [54, 54, 54], # 黑色
	'page': [159, 121, 238], # 紫色
	'table': [255, 99, 71], # 红色
	'picture': [255, 236, 139], # 黄色
	'cell': [255, 222, 173], #橙色
	} 

def load_table_json(maindir):
	contents_dict = {}
	dirlist = os.listdir(os.path.join(maindir))
	for idx, docid in enumerate(dirlist):
		# docid = '169'
		if docid == 'errors':
			continue
		contents_dict[docid] = {}
		with open(os.path.join(maindir, docid, 'pages_with_tables'), 'r') as fo:
			data = json.load(fo)
			n_processed = 0
			for pageid in data:
				n_processed += 1
				logging.info('Read Json Files: doc: %s, doc rate: %.2f%%, page: %s, page rate: %.2f%%' % (
					docid, 100.0 * (idx+1) / len(dirlist), 
					pageid, 100.0 * (n_processed) / len(data)))
				contents_dict[docid][pageid] = []
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
				# 将表格框中的短线合并为长线
				for table in tables:
					# 合并内线
					finished, old_lines = False, table['lines']
					while not finished:
						line_used = [False] * len(old_lines)
						new_lines = []
						finished = True
						for i in range(len(old_lines)):
							if not line_used[i]:
								new_line = old_lines[i]
								merged = False
								for j in range(i+1, len(old_lines)):
									[line1, line2, merged] = _merge_two_lines(
										old_lines[i]['position'], old_lines[j]['position'])
									if merged:
										new_line = {
											'position': line1, 'type': old_lines[i]['type']}
										line_used[j] = True
										break
								new_lines.append(new_line)
								if merged:
									finished = False
						old_lines = new_lines
					lines = old_lines
					table['lines'] = old_lines
					points = []
					# 计算出交点
					for i in range(len(old_lines)):
						for j in range(i+1, len(old_lines)):
							point = _cal_line_intersection(
								old_lines[i]['position'], old_lines[j]['position'])
							if point:
								points.append(point)
					# 去重
					for x in points:
						while points.count(x) > 1:
							del points[points.index(x)]
					# 先按照x坐标顺序，再按照y坐标顺序排列
					new_points, point = [], []
					points = sorted(points, key=lambda x: x[0])
					for p in points:
						if not point or p[0] == point[-1][0]:
							point.append(p)
						elif point:
							new_points.append(point)
							point = [p]
					if point:
						new_points.append(point)
					row_points = {}
					for i in range(len(new_points)):
						row_points[new_points[i][0][0]] = sorted(new_points[i], key=lambda x: x[1])
					# 先按照y坐标顺序，再按照z坐标顺序排列
					new_points, point = [], []
					points = sorted(points, key=lambda x: x[1])
					for p in points:
						if not point or p[1] == point[-1][1]:
							point.append(p)
						elif point:
							new_points.append(point)
							point = [p]
					if point:
						new_points.append(point)
					col_points = {}
					for i in range(len(new_points)):
						col_points[new_points[i][0][1]] = sorted(new_points[i], key=lambda x: x[0])
					# 按顺序遍历获得单元格
					cells = []
					for row in row_points:
						for i in range(len(row_points[row])):
							col = row_points[row][i][1]
							for j in range(i+1, len(row_points[row])):
								for k in range(len(col_points[col])):
									if col_points[col][k][0] > row:
										lt = row_points[row][i]
										lb = row_points[row][j]
										rt = col_points[col][k]
										rb = [rt[0], lb[1]]
										for l in range(len(col_points[rb[1]])):
											if col_points[rb[1]][l][0] == rb[0]:
												celled = _is_cell(lt, rt, lb, rb, lines)
												if celled:
													cells.append([lt[0], rb[0], lt[1], rb[1]])
												break
					# 去除其中较大的单元格
					for i in range(len(cells)):
						if _is_small_rect(cells[i], cells):
							table['cells'].append({'position': cells[i], 'type': 6})
						
				contents_dict[docid][pageid] = {
					'texts': texts, 'size': size, 'tables': tables,
					'others': others, 'curves': curves}
	
	return contents_dict

def _merge_two_lines(line1, line2):
	thresh = 5
	if line1[0] == line1[1] == line2[0] == line2[1]:
		if (line1[2] < line1[3] < line2[2] < line2[3] and line2[2] - line1[3] >= thresh) or \
			(line2[2] < line2[3] < line1[2] < line1[3] and line1[2] - line2[3] >= thresh):
			return line1, line2, False
		else:
			merged_line = [
				line1[0], line1[0], 
				min(line1[2], line2[2]), max(line1[3], line2[3])]
			return merged_line, None, True
	elif line1[2] == line1[3] == line2[2] == line2[3]:
		if (line1[0] < line1[1] < line2[0] < line2[1] and line2[0] - line1[1] >= thresh) or \
			(line2[0] < line2[1] < line1[0] < line1[1] and line2[0] - line1[1] >= thresh):
			return line1, line2, False
		else:
			merged_line = [
				min(line1[0], line2[0]), max(line1[1], line2[1]),
				line1[2], line1[2]]
			return merged_line, None, True
	else:
		return line1, line2, False
	
def _cal_line_intersection(line1, line2):
	thresh = 5
	if (line1[0] == line1[1] and line2[0] == line2[1]) or \
		(line1[2] == line1[3] and line2[2] == line2[3]):
		return None
	elif line1[0] == line1[1] and line2[2] == line2[3]:
		if (line1[2] - thresh) <= line2[2] <= (line1[3] + thresh) and \
			(line2[0] - thresh) <= line1[0] <= (line2[1] + thresh):
			return [line1[0], line2[2]]
	elif line2[0] == line2[1] and line1[2] == line1[3]:
		if (line2[2] - thresh) <= line1[2] <= (line2[3] + thresh) and \
			(line1[0] - thresh) <= line2[0] <= (line1[1] + thresh):
			return [line2[0], line1[2]]
		
def _is_cell(lt, rt, lb, rb, lines):
	thresh = 5
	rect_lines = []
	rect_lines.append([lt[0], lb[0], lt[1], lb[1]])
	rect_lines.append([lt[0], rt[0], lt[1], rt[1]])
	rect_lines.append([rt[0], rb[0], rt[1], rb[1]])
	rect_lines.append([lb[0], rb[0], lb[1], rb[1]])
	for line1 in rect_lines:
		is_intersect = False
		for l in lines:
			line2 = l['position']
			if (line1[0] == line1[1] and line2[0] == line2[1]):
				if line1[2] >= (line2[2] - thresh) and line1[3] <= (line2[3] + thresh):
					is_intersect = True
					break
			elif (line1[2] == line1[3] and line2[2] == line2[3]):
				if line1[0] >= (line2[0] - thresh) and line1[1] <= (line2[1] + thresh):
					is_intersect = True
					break
		if not is_intersect:
			return False
	return True

def _is_small_rect(rect1, cells):
	for rect2 in cells:
		if rect2[0] == rect1[0] and rect2[1] == rect1[1] and \
			rect2[2] == rect1[2] and rect2[3] == rect1[3]:
			continue
		if rect2[0] >= rect1[0] and rect2[1] <= rect1[1] and \
			rect2[2] >= rect1[2] and rect2[3] <= rect1[3]:
			return False
	return True

def write_json(contents_dict, path):
	with open(path, 'w') as fw:
		if 'Windows' in platform.platform():
			fw.write(json.dumps(contents_dict, indent=4))
		elif 'Linux' in platform.platform():
			fw.write(json.dumps(contents_dict))

if 'Windows' in platform.platform():
	contents_dict = load_table_json('E:\\Temporal\Python\darknet-master\datasets\\table-png\JPEGImages')
	write_json(contents_dict, 'E:\\Temporal\Python\darknet-master\datasets\\table-png\\texts.json')
elif 'Linux' in platform.platform():
	contents_dict = load_table_json('/home/wangxu/data/pdf2jpg_v4/output/')
	write_json(contents_dict, '/home/caory/github/table-detection/datasets/table-v2/texts.json')

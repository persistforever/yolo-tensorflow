# -*- coding: utf-8 -*-
# author: ronniecao
import sys
import platform
import codecs
import re

def read_txt(path, outpath):
	sentences = []
	pattern = re.compile(u'，|。|；|！|,|;')
	with codecs.open(path, 'r', 'utf8') as fo:
		for line in fo:
			line = line.strip()
			if len(line) >= 50:
				# print(line.encode('utf8'))
				for stc in pattern.split(line):
					sentences.append(stc)
	
	with open(outpath, 'w') as fw:
		for sentence in sentences:
			if sentence != '':
				fw.writelines((sentence + '\n').encode('utf8'))

if 'Windows' in platform.platform():
	sentences = read_txt('E:\\Github\\table-detection\\data\\table-v3\\sentences.txt', 
		'E:\\Github\\table-detection\\data\\table-v3\\new_sentences.txt')
elif 'Linux' in platform.platform():
	sentences = read_txt('/home/caory/github/table-detection/data/table-v2/texts.json')
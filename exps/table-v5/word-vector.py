# -*- coding: utf-8 -*-
# author: ronniecao
import sys
import platform
import codecs
import re
from pyltp import Segmentor
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy


def read_txt(path, outpath):
	sentences = []
	pattern = re.compile(u'，|。|；|！|,|;')
	with codecs.open(path, 'r', 'utf8') as fo:
		for line in fo:
			line = line.strip()
			if len(line) >= 25:
				# print(line.encode('utf8'))
				for stc in pattern.split(line):
					sentences.append(stc)
	
	with open(outpath, 'w') as fw:
		for sentence in sentences:
			if sentence != '':
				fw.writelines((sentence + '\n').encode('utf8'))

def split_words(path, outpath):
    segmentor = Segmentor()
    segmentor.load('E:\\Github\\table-detection\\data\\table-v3\\ltp_data\\cws.model')
    lines, sentences = [], []

    with codecs.open(path, 'r', 'utf8') as fo:
        for line in fo:
            lines.append(line.strip())

    for idx, line in enumerate(lines):
        print '%.4f%%' % (100.0 * idx / len(lines))
        words = segmentor.segment(line.encode('utf8'))
        sentence = [w.decode('utf8') for w in words]
        sentences.append(sentence)

    print(len(sentences))

    with open(outpath, 'w') as fw:
    	for sentence in sentences:
    		fw.writelines((' '.join(sentence) + '\n').encode('utf8'))

def word2vec(path, outpath, wordpath):
    sentences, word_dict = [], {}
    with codecs.open(path, 'r', 'utf8') as fo:
        for line in fo.readlines():
            line = line.strip().split(' ')
            sentences.append(line)
    
    print('Start Training ...')
    model = gensim.models.Word2Vec(sentences, size=100, iter=200, min_count=20, sg=1)

    word_vector, word_dict, n = [], {}, 0
    with open(outpath, 'w') as fw:
        for word in model.wv.index2word:
            word_dict[n] = word
            n += 1
            word_vector.append(model[word])
            fw.writelines((word + '\t' + ' '.join([str(t) for t in model[word]]) + '\n').encode('utf8'))

    with open(wordpath, 'w') as fw:
        for word in model.wv.index2word:
            fw.writelines((word + '\n').encode('utf8'))

    drawing(numpy.array(word_vector), word_dict)

def drawing(vectorpath):
    word_vector, word_dict, n = [], {}, 0
    with codecs.open(vectorpath, 'r', 'utf8') as fo:
        for line in fo:
            [word, vector] = line.strip().split('\t')
            vector = [float(t) for t in vector.split(' ')]
            word_dict[n] = word
            n += 1
            word_vector.append(vector)
    word_vector = numpy.array(word_vector)

    tsne = TSNE(n_components=2)
    tsne.fit(word_vector[0:1000, :])
    word_embedding = tsne.embedding_
    print word_embedding.shape
    fig = plt.figure()
    for idx in range(word_embedding.shape[0]) :
        plt.plot(word_embedding[idx,0], word_embedding[idx,1], 'o-', color='#ef4136')
        plt.text(word_embedding[idx,0], word_embedding[idx,1], word_dict[idx], color='black', ha='left')
    plt.show()

if 'Windows' in platform.platform():
    # sentences = read_txt('E:\\Github\\table-detection\\data\\table-v5\\sentences.txt', 
    #     'E:\\Github\\table-detection\\data\\table-v5\\new_sentences.txt')
    # split_words('E:\\Github\\table-detection\\data\\table-v5\\new_sentences.txt', \
    #     'E:\\Github\\table-detection\\data\\table-v5\\split_sentences.txt')
    # word2vec('E:\\Github\\table-detection\\data\\table-v5\\split_sentences.txt', \
    #     'E:\\Github\\table-detection\\data\\table-v5\\word_vector.txt',
    #     'E:\\Github\\table-detection\\data\\table-v5\\word_dict.txt')
    drawing('E:\\Github\\table-detection\\data\\table-v5\\word_vector.txt')
elif 'Linux' in platform.platform():
    sentences = read_txt('/home/caory/github/table-detection/data/table-v2/texts.json')

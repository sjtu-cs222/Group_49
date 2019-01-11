# -*- coding: utf-8  -*-
import multiprocessing
import json
import codecs
import jieba
import pickle
# from opencc import OpenCC
# from gensim.models import Word2Vec
# from gensim.models.word2vec import LineSentence
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(funcName)s - %(message)s')


def create_data(file):
	logging.info("Creating Dataset...")
	examples = []
	for line in codecs.open(file,'rb',encoding='utf8'):
		# print(line)
		item = json.loads(line)
		examples.append(item['comment'])

	return examples

def preprossessing(file):
	add_punc = '，。、【 】 “”：；（）《》‘’？！⑦()、%^>℃：.”“^-——=&#@￥'
	data = []
	vocabs = []
	mapping = {}
	cnt = 0

	for line in codecs.open('intent2id.txt','rb',encoding='utf8'):
		items = line.split()
		intent = items[0]
		intentID = items[1]
		mapping[intentID] = intent

	for line in codecs.open(file,'rb',encoding='utf8'):
		cnt += 1
		args = line.split()
		text = ""
		try:
			label = args[-2].split("-")[1]
			for i in range(len(args)-2):
				words = list(jieba.cut(args[i]))
				for word in words:
					if word not in add_punc:
						text += (" " + word)
			vocabs.extend(words)
			# print(text)
			data.append(text)
			data.append('===>' + mapping[label]+'\n')
		except:
			continue
	
	# vocabs = set(vocabs)
	# print(len(vocabs))
	# print(vocabs)
	with codecs.open("seg.txt",'w','utf8') as f:
		for word in data:
			f.write(word)

	# with open('rubbish.pkl') as f:
	# 	pickle.dump(f,data)
	return data

def train(data):

	outdir = "leyan.model"
	model = Word2Vec([s.split() for s in data],size=300,
			window=5,min_count=1,
			workers=multiprocessing.cpu_count()-1)

	model.save(outdir)

if __name__ == '__main__':
	# data = preprossessing('yanjing.anonymous.replace.txt')
	# train(data)
	# train('test.json')
	# train('items.json')
	# x = create_data('test.json')
	# preprossessing(x)
	# data = preprossessing(x)

from hyperparams import Hyperparams as hp
import os
from collections import Counter
import regex
import numpy as np
import tensorflow as tf





def load_vocab(fname):
	"""
	加载词典
	:param fname:
	:return:
	"""
	vocab  = [line.split()[0] for line in open(fname,'r').readlines() if int(line.split()[1]) > hp.min_cnt]
	word2idx = {word : idx for idx, word in enumerate(vocab)}
	idx2word = {idx : word for idx, word in enumerate(vocab)}
	return  word2idx,idx2word


def make_vocab(fpath, fname):
	'''
	处理原始数据,对词计数,然后按词频从大到小,每行保存。
	单词 频率
	the	155830
	to	97230
	of	91435
	and	84609
	:param fpath: 原始文件路径
	:param fname: 输出文件
	:return:
	'''
	with open(fpath, 'r') as f:
		# words = [regex.sub("[^\s\p{Latin}']", "", line).split() for line in f.readlines()]
		text = f.read()
		text = regex.sub("[^\s\p{Latin}']", "", text)
		words = text.split()
		word2cnt = Counter(words)
		with open(fname,'w') as fout:
			fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>","</S>"))
			for word, cnt in word2cnt.most_common(len(word2cnt)):
				fout.write("{}\t{}\n".format(word, cnt))


def create_data(source_sent, target_sent):
	de2idx, idx2de = load_vocab(hp.de_processed_fname)
	en2idx, idx2en = load_vocab(hp.en_processed_fname)
	x_list, y_list , Sources, Targets = [], [], [], []
	for source_sent, target_sent in zip(source_sent, target_sent):
		x = [de2idx.get(word, 1) for word in (source_sent + " </S>").split()]
		y = [en2idx.get(word, 1) for word in (target_sent + " </S>").split()]
		if max(len(x), len(y)) < hp.maxlen:
			x_list.append(np.array(x))
			y_list.append(np.array(y))
			Sources.append(source_sent)
			Targets.append(target_sent)
	
	X = np.zeros([len(x_list), hp.maxlen], np.int32)
	Y = np.zeros([len(y_list), hp.maxlen], np.int32)
	for i, (x, y) in enumerate(zip(x_list, y_list)):
		X[i] = np.pad(x, (0, hp.maxlen - len(x)), 'constant', constant_values=(0, 0))
		Y[i] = np.pad(y, (0, hp.maxlen - len(y)), 'constant', constant_values=(0, 0))
	
	return X, Y, Sources, Targets


def load_train_data():
	de_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in open(hp.source_train,'r').readlines() if line[0] !="<"]
	en_sents = [regex.sub("[^\s\p{Latin}']", "", line) for line in open(hp.target_train, 'r').readlines() if line[0] !="<"]
	
	X, Y, Sources, Targets = create_data(de_sents, en_sents)
	
	return X, Y

def get_batch_data():
	# Load train data
	X, Y = load_train_data()
	num_batch = len(X) // hp.batch_size
	
	# convert to tensor
	
	X = tf.convert_to_tensor(X, tf.int32)
	Y = tf.convert_to_tensor(Y, tf.int32)
	
	# create Queues
	input_queues = tf.train.slice_input_producer([X, Y])
	
	# create batch queues
	x, y = tf.train.shuffle_batch(input_queues, num_threads=8, batch_size=hp.batch_size, capacity=hp.batch_size*64,
	                              min_after_dequeue=hp.batch_size*32,
	                              allow_smaller_final_batch=False)
	return x, y, num_batch
	
	
if __name__  == '__main__':
	make_vocab(hp.source_train, hp.de_processed_fname)
	make_vocab(hp.target_train, hp.en_processed_fname)
	print("Done")
	
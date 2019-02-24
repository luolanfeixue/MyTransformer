
from data_core import load_vocab,get_batch_data
from hyperparams import Hyperparams as hp
import tensorflow as tf
from tqdm import tqdm

class Graph():
	def __init__(self,is_training = True):
		self.graph = tf.Graph()
		with self.graph.as_default():
			if is_training:
				self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
				# self.x, self.y, self.num_batch = None, None, None  # (N, T)
			else:  # inference
				self.x = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
				self.y = tf.placeholder(tf.int32, shape=(None, hp.maxlen))
				
			# define decoder inputs
			self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)
		
			de2idx, idx2de = load_vocab(hp.de_processed_fname)
			en2idx, idx2en = load_vocab(hp.en_processed_fname)
	


if __name__ == '__main__':
	de2idx,idx2de = load_vocab(hp.de_processed_fname)
	en2idx,idx2en = load_vocab(hp.en_processed_fname)
	
	# build graph
	g = Graph()
	print("Graph Loaded")
	
	# start session
	sv = tf.train.Supervisor(graph=g.graph, logdir=hp.logdir, save_model_secs=0)
	
	with sv.managed_session() as sess:
		for epoch in range(1, hp.num_epochs + 1):
			if sv.should_stop(): break
			for step in tqdm(range(g.num_batch), total = g.num_batch, ncols = 70, leave = False, unit = 'b'):
				sess.run(g.decoder_inputs)
				
			# gs = sess.run(g.global_step)
			sv.saver.save(sess, hp.logdir + '/model_epoch_%02d_gs_%d' % (epoch, gs))
			
	print("Done")


class Hyperparams:
	source_train = 'corpora/train.tags.de-en.de'
	target_train = 'corpora/train.tags.de-en.en'
	# processed = 'preprocessed'
	de_processed_fname = 'preprocessed/de.vocab.tsv'
	en_processed_fname = 'preprocessed/en.vocab.tsv'
	source_test = 'corpora/IWSLT16.TED.tst2014.de-en.de.xml'
	target_test = 'corpora/IWSLT16.TED.tst2014.de-en.en.xml'

	
	
	# training
	batch_size = 32 # alias N
	lr = 0.0001
	logdir = 'logdir'
	
	
	#model
	maxlen = 10 # maxinum number of words in a sentence. alias = T
	
	min_cnt = 20 # words whose occured less than min_cnt are encoded as <UNK>
	hidden_units = 512 # alias = C
	num_blocks = 6 # number of encoder/decoder block
	num_epochs = 20
	num_heads = 8
	dropout_rate = 0.5
	sinusoid = False # If True, use sinusoid . If false, positional embedding.
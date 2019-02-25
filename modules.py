import tensorflow as tf
import numpy as np


def embedding(inputs, vocab_size, num_units, zero_pad=True, scale=True,
              scope="embedding", reuse=None):
    """
	
	:param inputs: 一个tensor，包含了词到id，以便于 lookup table
	:param vocab_size: 词典大小
	:param num_units: embeding 的 hidden units
	:param zero_pad: boolean, 如果是true,则第一行全为0
	:param scale: boolean,是否归一化，如果true，则全都除以 sqrt num_units
	:param scope: variable_scope
	:param reuse: boolean, 是否以相同的名字重新使用前一个layer的权重
	:return:
		被 embedding后的 tensor, 最后一个维度为num_units
    >>
    outputs.shape (2, 3, 9)
	"""

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table', dtype=tf.float32, shape=[vocab_size, num_units],
                                       initializer=tf.contrib.layers.xavier_initializer())
        #  lookup_table的第一行强制设置为0
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units])), lookup_table[1:, :], 0)
        # inputs.shape(32,10) lookup_table.shape=(vocab_size,512),outputs.shape(32,10,512)
        # inputs里存的都是词的id，有vocab_size个。
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)

    return outputs


# inputs = tf.to_int32(tf.reshape(tf.range(320), (32, 10)))
# outputs = embedding(inputs, 100000, 512, zero_pad=False)
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print(sess.run(outputs)) outputs.shape = (32,10,512)

def positional_encoding(inputs, num_units, zero_pad=True, scale=True, scope="positional_encoding", reuse=None):
    """
	:param inputs:  A 2d Tensor with shape of (N, T). N = batch_size, T = length of sequence
	:param num_units:
	:param zero_pad: boolean. If True, all the values of the first row ( id = 0) should be constant zero
	:param scale: boolean.  If True, the output will be multiplied by sqrt num_units(check details from paper)
	:param scope: Optional scope for `variable_scope`.
	:param reuse:  Boolean, whether to reuse the weights of a previous laye  by the same name.
	:return:  A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
	"""

    N, T = inputs.get_shape().as_list()  # N = 32, T = 10
    with tf.variable_scope(scope, reuse=reuse):
		# tf.range(T) 创建一个长度为T的序 shape = (10) 内容（0，1，2，。。。，9）
		# tf.expand_dims(tf.range(T), 0)  增加一个个纬度 shape = (1,10)
        # tf.tile (N,1) 第一个纬度复制N遍，shape(N,T) 内容（0，1, 2,....,9) 复制N行
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])
        # position_enc.shape = (10,512)
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])
        # seconde part, apply the cosine to even colums ans sin to odds
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.sin(position_enc[:, 1::2])  # dim 2i+1

        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units])), lookup_table[1:, :], 0)
        # lookup_table.shape =(10,512) position_ind.shape=(32,10)
        # lookup_table的第一个纬度是position_ind中内容id的个数。其id为（0-9）
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * (num_units ** 0.5)
        return outputs



# inputs = tf.to_int32(tf.reshape(tf.range(320), (32, 10)))
# outputs = positional_encoding(inputs, 512, zero_pad=False)
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print(sess.run(outputs)) #outputs.shape (32,10,512)
# 	pass

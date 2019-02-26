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

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

	Args:
	  inputs: A tensor with 2 or more dimensions, where the first dimension has
		`batch_size`.
	  epsilon: A floating number. A very small number for preventing ZeroDivision Error.
	  scope: Optional scope for `variable_scope`.
	  reuse: Boolean, whether to reuse the weights of a previous layer
		by the same name.

	Returns:
	  A tensor with the same shape and data dtype as `inputs`.
	'''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries
                        , keys
                        , num_units=None
                        , num_head=0
                        , dropout_rate=0
                        , is_training=True
                        , causality=False
                        , scope="Nultihead_attention"
                        , reuse=None):
    """
	
	:param queries: a 3d tensor with shape of [N_q, T_q, C_q]
	:param keys: a 3d tensor with shape of [N_q, T_k, C_k]
	:param num_units: Attention size
	:param num_head: 多头个数，默认8
	:param dropout_rate:
	:param is_training:
	:param causality: if true, units that reference the future are masked
	:param scope:
	:param reuse:
	:return:
		a 3d tensor with shape of (N, T_q, C_q)
	"""

    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]
        # Linear projections
        # dense 实现的操作，outputs = activation(inputs * kernel + bias)
        # kernel是由层创建的权重矩阵,kernel.shape=(inputs.shape[1],num_units),inputs为输入数据，units 输出空间维度
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        # split and concat
        # split，axis=2的维度上分解为8个，然后在在axis=0的维度上合并。
        Q_ = tf.concat(tf.split(Q, num_head, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_head, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_head, axis=2), axis=0)  # (h*N, T_v, C/h)

        # Multiplication
        attention = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q,T_k)

        # scale
        attention = attention / (K_.get_shape().as_list()[-1] ** 0.5)

        # key中一整列为0的时候就，最终得出的outputs被一个极小数代替。
        # Key Masking
        # 最后一个维度求和,则维度变为(N,T_k)，之后求绝对值，（N,T_k)只包含正数和零。经过sign(符合化）变为0，1。
        # 0代表原来第三个维度所有值都为0，反之为1。那些为0的就是我们要mask的key
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k) 是否为正
        key_masks = tf.tile(key_masks, [num_head, 1])  # (h*N, T_k) 复制h遍
        # 在axis=1,增加一个维度，然后在这个维度复制T_q遍。第三个维度记录着我们要mask的key
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(attention) * (-2 ** 32 + 1)  # shape 与outputs相同，内容都是1，然后在乘一个极小数。内容就都是极小数
        # where(condition,x, y)。condition为true的地方保留x相应位置的元素，为false的时候保留为y相应位置的元素。
        # key_masks 为0的时候用一个极小数代替（padding中），否则时原来的outputs
        attention = tf.where(tf.equal(key_masks, 0), paddings, attention)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        # 只计算query长度都内的key,超出query长度的不计算attention
        if causality:
            diag_vals = tf.ones_like(attention[0, :, :])  # (T_q, T_k)
            # 下三角阵，其他位置都为0。这样对于每一个T_q,凡是那些大于它角标的T_k值全都为0
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape[attention][0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            # 用下三角取mask
            attention = tf.where(tf.equal(masks, 0), paddings, attention)  # (h*N, T_q, T_k)

        # Activation
        attention = tf.nn.softmax(attention)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-11)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_units, 1])  # (h*N,T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        attention *= query_masks  # broadcasting (N, T_q, C)

        # Dropouts
        attention = tf.layers.dropout(attention, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        outputs = tf.matmul(attention, V_)  # (h*N, T_q, C/h)
        outputs = tf.concat(tf.split(outputs, num_head, axis=0), axis=2)  # (N, T_q, C)
        outputs += queries
        outputs = normalize(outputs)  # (N,T_q, C)

        return outputs


#
# inputs = tf.to_int32(tf.reshape(tf.range(320), (32, 10)))
# query = embedding(inputs, 512, zero_pad=True)
# keys =  embedding(inputs, 512, zero_pad=True)
# outputs = multihead_attention(query,keys,causality=True)
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print(sess.run(outputs)) #outputs.shape (32,10,512)
# 	pass


def feedforward(inputs,
                num_units=[2048, 512],
                scope='multihead_attention',
                reuse=None):
    """
	
	:param inputs:A 3d tensor with shape of [N, T, C].
	:param num_units:
	:param scope:
	:param reuse:
	:return:
	"""
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters:": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}

        # 正常的图片是四个纬度（N, H, W, C),H代表高，W代表宽，C代表通道。
        # 这里 不是图片是一个纬度序列，用T代表，是一维的，C还是代表通道。
        # filter的size（1,1,C,num_units[0]) 当制定conv1d，前三个纬度就已经确定，只需指定filter的数量，即第三个纬度。
        outputs = tf.layers.conv1d(**params) # （N, T, num_units[0])

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(*params) # （N, T, num_units[1])

        outputs += inputs
        outputs = normalize(outputs)

    return outputs

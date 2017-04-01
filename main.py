import numpy as np
import pandas as pd
import tensorflow as tf

def weigths_initializer(shape):
	return tf.Variable(tf.random_normal(shape, stddev=0.01))

def reshape(x):
	x2 = np.zeros((x.shape[0], x.shape[1]+1))
	x2[:,1:] = x
	x2[:, 0] = 1
	return x2
# def softmax(z):
# 	expz = np.exp(z)
# 	return expz / expz.sum(axis=1, keepdims=True)

# def forward(x, y, w, v):
# 	x2 = reshape(x)
# 	a = x2.dot(w)
# 	z = 1 / (1 + exp(-a))
# 	z2 = reshape(z)
# 	a2 = z2.dot(v)
# 	h = softmax(a2)
# 	return h, z
# def classification_rate(y, p):


def forward(x, w, v):
	z = tf.nn.sigmoid(tf.matmul(reshape(x), w))
	return tf.matmul(reshape(z), v)

def main():
	D = 2
	M = 3
	N = 500

	X1 = np.random.randn(N, D) + np.array([-2, 2])
	X2 = np.random.randn(N, D) + np.array([2, 2])
	X3 = np.random.randn(N, D) + np.array([0, -2])
	X = np.vstack([X1, X2, X3])

	Y = np.array([0]*N + [1]*N + [2]*N)
	K = len(set(Y))
	N = len(Y)
	T = np.zeros((N, K))
	for n in range(N):
		T[n, Y[n]] = 1

	tfx = tf.placeholder(tf.float32, [None, D])
	tfy = tf.placeholder(tf.float32, [None, K])

	w = weigths_initializer([D+1, M])
	v = weigths_initializer([M+1, K])

	p = forward(tfx, w, v)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(p, tfy))
	train_op = tf.train.GradienDescentOptimizer(0.05).minimize(cost)
	predict_op = tf.argmax(p, 1)
	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	for i in xrange(1000):
		sess.run(train_op, feed_dict={tfx: X, tfy: Y})
		pred = sess.run(predict_op, feed_dict={tfx: X, tfy: Y})
		print np.mean(Y == pred)


if __name__ == '__main__':
	main()
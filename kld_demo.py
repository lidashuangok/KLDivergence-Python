# !usr/bin/python
# Author:das
# -*-coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from scipy.stats import norm

x = np.arange(-10,10,0.001)
pdf = norm.pdf(x,0,2).reshape(1,-1)
# print(np.zeros(1))
p = tf.placeholder(tf.float64,shape=pdf.shape)
mu = tf.Variable(np.zeros(1))
sigma = tf.Variable(np.eye(1))
normal = tf.exp(-tf.square(x - mu) / (2 * sigma))
q = normal / tf.reduce_sum(normal)

print(p)
print(q)

kl_loss = tf.reduce_sum(tf.where(p==0,tf.zeros(pdf.shape,tf.float64),p*tf.log(p/q)))


learn_rate = 0.0001
epoch = 100

opt = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(kl_loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epoch):
        sess.run(opt,feed_dict={p:pdf})
        print("%f loss is %f" % (i,sess.run(kl_loss,feed_dict={p:pdf})) )

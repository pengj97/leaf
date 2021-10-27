import tensorflow.compat.v1 as tf
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='/gpu:3'
tf.disable_eager_execution()

lr = 1
x = tf.Variable(1, dtype=tf.float32)

sess = tf.Session()
init = tf.global_variables_initializer()
with sess.as_default():
    sess.run(init)
    for i in range(10):
        lr = 1 / (i + 1)
        update = tf.assign(x, x - lr)

        sess.run(update)
        print(sess.run(x))

# lr = tf.placeholder(dtype=tf.float32)
# x = tf.Variable(1, dtype=tf.float32)
# update = tf.assign(x, x - lr)

# sess = tf.Session()
# init = tf.global_variables_initializer()
# with sess.as_default():
#     sess.run(init)
#     for i in range(10):
#         l = 1 / (i + 1)
#         sess.run(update, feed_dict={lr:l})
#         print(sess.run(x))
import tensorflow.compat.v1 as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='/gpu:3'
tf.compat.v1.disable_eager_execution()

y = tf.placeholder(dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w = tf.Variable(2,dtype=tf.float32)
#prediction
 
#define losses3
l = tf.square(w * x - y)
g = tf.gradients(l, w)
learning_rate = tf.constant(1,dtype=tf.float32)
#learning_rate = tf.constant(0.11,dtype=tf.float32)
init = tf.global_variables_initializer()
 
#update
update = tf.assign(w, w - learning_rate * g[0])
 
with tf.Session() as sess:
    sess.run(init)
    #print(sess.run([g,w*x,w], {x: 1, y:3}))
    for _ in range(5):
        w_, g_,  l_ = sess.run([w,g,l],feed_dict={x:1, y:3})
        print('variable is w:',w_, ' g is ',g_,'  and the loss is ',l_)

        _ = sess.run(update,feed_dict={x:1, y:3})

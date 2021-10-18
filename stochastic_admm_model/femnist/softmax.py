import tensorflow.compat.v1 as tf

from model import Model
import numpy as np


IMAGE_SIZE = 28


class ClientModel(Model):
    def __init__(self, seed, lr, num_classes):
        self.num_classes = num_classes
        super(ClientModel, self).__init__(seed, lr)

    def create_model(self):
        """Model function for CNN."""
        features = tf.placeholder(
            tf.float32, shape=[None, IMAGE_SIZE * IMAGE_SIZE], name='features')
        labels = tf.placeholder(tf.int64, shape=[None], name='labels')
        # softmax_regression : input_layer(784, 1) -> output_layer(62, 1)
        logits = tf.layers.dense(inputs=features, units=self.num_classes)
        predictions = {
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
        # TODO: Confirm that opt initialized once is ok?
        train_op = self.optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        eval_metric_ops = tf.count_nonzero(tf.equal(labels, predictions["classes"]))
        return features, labels, train_op, eval_metric_ops, loss

    def process_x(self, raw_x_batch):
        return np.array(raw_x_batch)

    def process_y(self, raw_y_batch):
        return np.array(raw_y_batch)

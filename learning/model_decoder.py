import numpy as np
import tensorflow as tf
from utils import reshape_cal_len

class ModelDE(object):

    def __init__(self, precision):
        self.precision = precision

    def forward_(self, fea_input, recos_target, is_training):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            self.fea_input = fea_input

            self.is_training = is_training

            print('self.fea_input', self.fea_input)
            x, dims = reshape_cal_len(self.fea_input)

            W_conv1 = self._weight_variable([dims, 1024], scope='w1')
            b_conv1 = self._bias_variable([1024], scope='b1')
            h_conv1 = tf.nn.xw_plus_b(x, W_conv1, b_conv1)
            h_conv1 = self._batch_norm('bn11', h_conv1)
            h_conv1 = tf.nn.relu(h_conv1)

            W_conv11 = self._weight_variable([1024, 1024], scope='conv_w11')
            b_conv11 = self._bias_variable([1024], scope='b11')
            h_conv1 = tf.nn.xw_plus_b(h_conv1, W_conv11, b_conv11)
            h_conv1 = self._batch_norm('bn12', h_conv1)
            h_conv1 = tf.nn.relu(h_conv1)

            W_fc2 = self._weight_variable([1024, 3*32*32], scope='fcw2')
            b_fc2 = self._bias_variable([3*32*32], scope='fcb2')

            x_last = tf.matmul(h_conv1, W_fc2) + b_fc2

            output = tf.reshape(x_last, [-1, 32, 32, 3])

            print('recos_target shape', recos_target.shape)

            recos_target_norm = recos_target/255-0.5
            loss = tf.reduce_mean((output - recos_target_norm) ** 2)

        return loss, output, recos_target_norm

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(
                inputs=x,
                decay=.9,
                center=True,
                scale=True,
                activation_fn=None,
                updates_collections=None,
                is_training=self.is_training)

    def _weight_variable(self, shape, scope):
        with tf.variable_scope(scope):
            w = tf.get_variable('DW', dtype=self.precision, initializer=tf.truncated_normal(shape, stddev=0.1,
                                                                                            dtype=self.precision))  # TODO: init is a constant
        return w

    def _bias_variable(self, out_dim, scope):
        with tf.variable_scope(scope):
            b = tf.get_variable('biases', dtype=self.precision,
                                initializer=tf.constant(0.1, shape=[out_dim[0]], dtype=self.precision))
        return b


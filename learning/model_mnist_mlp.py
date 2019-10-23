import numpy as np
import tensorflow as tf


class ModelMNISTMLP(object):
  def __init__(self, layer_used=[1, 3, 4], ratio=0.1, precision=tf.float32, architecture='MLP', label_smoothing=0):
      self.architecture = architecture
      self.layer_used = layer_used
      self.precision = precision
      self.ratio = ratio
      self.label_smoothing = label_smoothing
      print("using MLP")
      # self.discriminator(self._encoder())

  # def _encoder_v2(self, x_input, is_train):
  #     with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
  #       self.x_input = x_input
  #       self.is_training = is_train
  #
  #       self.x_image = tf.reshape(self.x_input, [-1, 28* 28])
  #
  #       # first convolutional layer
  #       W_conv1 = self._weight_variable([28*28, 1024], scope='w1')
  #       b_conv1 = self._bias_variable([1024], scope='b1')
  #       h_conv1 = self._conv2d(self.x_image, W_conv1) + b_conv1
  #       h_conv1 = self._batch_norm('bn11', h_conv1)  #TODO: previous mnist may have bug in naming the bn layer.
  #       self.x1 = h_conv1
  #       h_conv1 = tf.nn.relu(h_conv1)
  #
  #       W_conv11 = self._weight_variable([3, 3, 32, 64], scope='conv_w11')
  #       b_conv11 = self._bias_variable([64], scope='b11')
  #       h_conv1 = self._conv2d(h_conv1, W_conv11) + b_conv11
  #       h_conv1 = self._batch_norm('bn12', h_conv1)
  #       self.x1 = h_conv1
  #       h_conv1 = tf.nn.relu(h_conv1)
  #
  #       h_pool1 = self._max_pool_2x2(h_conv1)
  #
  #
  #
  #       # second convolutional layer
  #       W_conv2 = self._weight_variable([3,3,64,128], scope='conv_w2')
  #       b_conv2 = self._bias_variable([128], scope='b2')
  #       h_conv2 = self._conv2d(h_pool1, W_conv2) + b_conv2
  #       h_conv2 = self._batch_norm('bn2', h_conv2)
  #       self.x2 = h_conv2
  #       h_conv2 = tf.nn.relu(h_conv2)
  #
  #
  #       W_conv21 = self._weight_variable([3, 3, 128, 256], scope='conv_w21')
  #       b_conv21 = self._bias_variable([256], scope='b21')
  #       h_conv2 = self._conv2d(h_conv2, W_conv21) + b_conv21
  #       h_conv2 = self._batch_norm('bn21', h_conv2)
  #       self.x2 = h_conv2
  #       h_conv2 = tf.nn.relu(h_conv2)
  #
  #
  #
  #       h_pool2 = self._max_pool_2x2(h_conv2)
  #
  #       # first fully connected layer
  #       W_fc1 = self._weight_variable([7 * 7 * 256, 1024], scope='fcw1')
  #       b_fc1 = self._bias_variable([1024], scope='fcb1')
  #
  #       h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 256])
  #       h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
  #       h_fc1 = self._batch_norm('bnfc', h_fc1)
  #       # self.x3 = h_fc1  #TODO: bug
  #       h_fc1 = tf.nn.relu(h_fc1)
  #
  #       self.x3 = h_fc1
  #
  #       # output layer
  #       W_fc2 = self._weight_variable([1024, 10], scope='fcw2')
  #       b_fc2 = self._bias_variable([10], scope='fcb2')
  #
  #       self.pre_softmax = tf.matmul(h_fc1, W_fc2) + b_fc2
  #
  #       return self.pre_softmax, self.x3

  def _encoder(self, x_input, y_in, is_train, mask_effective_attack=False):
      with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
        self.x_input = x_input
        self.y_input = y_in
        self.is_training = is_train

        self.x_image = tf.reshape(self.x_input, [-1, 28* 28])

        # first convolutional layer
        W_conv1 = self._weight_variable([28*28, 1024], scope='w1')
        b_conv1 = self._bias_variable([1024], scope='b1')
        # h_conv1 = self._conv2d(self.x_image, W_conv1) + b_conv1
        h_conv1 = tf.nn.xw_plus_b(self.x_image, W_conv1, b_conv1)
        h_conv1 = self._batch_norm('bn11', h_conv1)  #TODO: previous mnist may have bug in naming the bn layer.
        self.x1 = h_conv1
        h_conv1 = tf.nn.relu(h_conv1)

        W_conv11 = self._weight_variable([1024, 1024], scope='conv_w11')
        b_conv11 = self._bias_variable([1024], scope='b11')
        # h_conv1 = self._conv2d(h_conv1, W_conv11) + b_conv11
        h_conv1 = tf.nn.xw_plus_b(h_conv1, W_conv11, b_conv11)
        h_conv1 = self._batch_norm('bn12', h_conv1)
        self.x1 = h_conv1
        h_conv1 = tf.nn.relu(h_conv1)

        W_fc1 = self._weight_variable([1024, 1024], scope='fcw1')
        b_fc1 = self._bias_variable([1024], scope='fcb1')
        h_conv1 = tf.matmul(h_conv1, W_fc1) + b_fc1
        h_conv1 = self._batch_norm('bn12', h_conv1)
        h_conv1 = tf.nn.relu(h_conv1)

        # h_pool1 = self._max_pool_2x2(h_conv1)

        # W_fc1 = self._weight_variable([7 * 7 * 256, 1024], scope='fcw1')
        # b_fc1 = self._bias_variable([1024], scope='fcb1')
        #
        # h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 256])
        # h_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
        # h_fc1 = self._batch_norm('bnfc', h_fc1)
        # # self.x3 = h_fc1  #TODO: bug
        # h_fc1 = tf.nn.relu(h_fc1)
        #
        self.x2 = None
        self.x3 = h_conv1

        # output layer
        W_fc2 = self._weight_variable([1024, 10], scope='fcw2')
        b_fc2 = self._bias_variable([10], scope='fcb2')

        self.pre_softmax = tf.matmul(h_conv1, W_fc2) + b_fc2

        # y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #     labels=self.y_input, logits=

        ce_labels = tf.one_hot(self.y_input, 10)
        y_xent = tf.losses.softmax_cross_entropy(
            onehot_labels=ce_labels,
            logits=self.pre_softmax,
            label_smoothing=self.label_smoothing)

        self.y_xent = y_xent

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        self.weight_decay_loss = self._decay()

        self.correct_prediction = tf.equal(self.y_pred, self.y_input)

        mask = tf.cast(self.correct_prediction, tf.float32)

        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        if mask_effective_attack:
            mask_temp = tf.expand_dims(mask, 1)
            raw_xent = y_xent * mask_temp * self.ratio + y_xent * (1 - mask_temp) * (1 - self.ratio)  ##TODO:potential bug here
            self.xent = tf.reduce_sum(raw_xent)
            self.mean_xent = tf.reduce_mean(raw_xent)

        else:
            self.xent = tf.reduce_sum(y_xent)
            # self.mean_xent = self.reduce_sum_det(y_xent) / tf.cast(tf.shape(y_xent)[0], dtype=self.precision)
            self.mean_xent = tf.reduce_mean(y_xent)

        x0 = None
        x1 = None
        x2 = self.x1
        x3 = self.x2
        x4 = self.x3
        pre_softmax = self.pre_softmax
        xent = self.xent
        mean_xent = self.mean_xent
        weight_decay_loss = self.weight_decay_loss
        num_correct = self.num_correct
        accuracy = self.accuracy
        predictions = self.y_pred
        mask = mask

        layer_values = {'x0':x0, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4, 'pre_softmax':pre_softmax,
                        'softmax': tf.nn.softmax(pre_softmax)}

        return [layer_values, xent, mean_xent, weight_decay_loss, num_correct, accuracy, predictions, mask]

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

  def reduce_sum_det(self, x):
      v = tf.reshape(x, [1, -1])
      return tf.reshape(tf.matmul(v, tf.ones_like(v, dtype=self.precision), transpose_b=True), [])

  def _weight_variable(self, shape, scope):
      with tf.variable_scope(scope):
          w = tf.get_variable('DW', dtype=self.precision, initializer=tf.truncated_normal(shape, stddev=0.1, dtype=self.precision))  #TODO: init is a constant
      return w

  def _bias_variable(self, out_dim, scope):
      with tf.variable_scope(scope):
        b = tf.get_variable('biases', dtype=self.precision,
                          initializer= tf.constant(0.1, shape = [out_dim[0]], dtype=self.precision))
      return b

  def _conv2d(self, x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  def _max_pool_2x2(self, x):
      return tf.nn.max_pool(x,
                            ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1],
                            padding='SAME')
  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)


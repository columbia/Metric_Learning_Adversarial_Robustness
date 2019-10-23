import numpy as np
import tensorflow as tf


class CifarResNet(object):
  """ResNet model."""

  def __init__(self, precision=tf.float32, ratio=0.1, mode='20'):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.precision = precision
    self.ratio = ratio
    if mode == '50':
        self.block_list = [2, 3, 5, 2]
        print("using Res50")
    elif mode == '20':
        self.block_list = [1,1,1,1]
        print("using Res20")
    else:
        raise("please set res net type")


  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]


  def _encoder(self, x_input, y_in, is_train, mask_effective_attack=False):
    """Build the core model within the graph."""

    with tf.variable_scope('main_encoder', reuse=tf.AUTO_REUSE):
        with tf.variable_scope('input'):

              self.x_input = x_input
              self.y_input = y_in
              self.is_training = is_train

              input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                                       self.x_input)
              x0 = self._conv('init_conv', input_standardized, 3, 3, 64, self._stride_arr(1))  #32  #TODO: I changed the stride from 2 to 1
              x0 = self._batch_norm('bn0', x0)
              x0 = self._relu(x0, 0)

        # https://arxiv.org/pdf/1605.07146v1.pdf
        self.filters = [64, 128, 256, 512]

        filters = self.filters

        with tf.variable_scope('unit_1_0'):
            x = self.conv_residual(x0, self.filters[0], self.filters[0], self.filters[0], self._stride_arr(1), False)

        for i in range(1, self.block_list[0]):
            with tf.variable_scope('unit_1_%d' % i):
                x = self.id_residual(x, self.filters[0], self.filters[0], self.filters[0], self._stride_arr(1), False)
        x1 = x
        with tf.variable_scope('unit_2_0'):
            x = self.conv_residual(x1, self.filters[0], self.filters[0], self.filters[1], self._stride_arr(2), False)  #16

        for i in range(1, self.block_list[1]):
          with tf.variable_scope('unit_2_%d' % i):
              x = self.id_residual(x, self.filters[1], self.filters[1], self.filters[1], self._stride_arr(1), False)

        x2 = x
        with tf.variable_scope('unit_3_0'):
            x = self.conv_residual(x2, self.filters[1], self.filters[1], self.filters[2], self._stride_arr(2), False)  # 4
        for i in range(1, self.block_list[2]):
          with tf.variable_scope('unit_3_%d' % i):
              x = self.id_residual(x, self.filters[2], self.filters[2], self.filters[2], self._stride_arr(1), False)

        x3 = x
        with tf.variable_scope('unit_4_0'):
            x = self.conv_residual(x3, self.filters[2], self.filters[3], self.filters[3], self._stride_arr(2), False)  # 2
        for i in range(1, self.block_list[3]):
          with tf.variable_scope('unit_4_%d' % i):
              x = self.id_residual(x, self.filters[3], self.filters[3], self.filters[3], self._stride_arr(1), False)

        with tf.variable_scope('unit_last'):
          x = self._batch_norm('final_bn', x)
          x = self._relu(x, 0)
          x = self._global_avg_pool(x)

        x4= x
        with tf.variable_scope('logit'):
          pre_softmax = self._fully_connected_final(x4, 10)

    predictions = tf.argmax(pre_softmax, 1)
    correct_prediction = tf.equal(predictions, self.y_input)

    mask = tf.cast(correct_prediction, tf.int64)

    num_correct = tf.reduce_sum(mask)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    with tf.variable_scope('costs'):
        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=pre_softmax, labels=self.y_input)
        self.y_xent = y_xent

        if mask_effective_attack:
            mask_temp = tf.expand_dims(mask, 1)
            raw_xent = y_xent * mask_temp * self.ratio + y_xent * (1-mask_temp) * (1 - self.ratio)
            xent = tf.reduce_sum(raw_xent, name='y_xent')
            mean_xent = tf.reduce_mean(raw_xent)

        else:
            xent = tf.reduce_sum(y_xent)
            # self.mean_xent = self.reduce_sum_det(y_xent) / tf.cast(tf.shape(y_xent)[0], dtype=self.precision)
            mean_xent = tf.reduce_mean(y_xent)

        weight_decay_loss = self._decay()

    layer_values = {'x0':x0, 'x1':x1, 'x2':x2, 'x3':x3, 'x4':x4, 'pre_softmax':pre_softmax,
                    'softmax': tf.nn.softmax(pre_softmax)}

    return [layer_values, xent, mean_xent, weight_decay_loss, num_correct, accuracy, predictions, mask]




  def match_loss(self, fea, loss_type, batchsize):
      fea1, fea2 = tf.split(fea, [batchsize, batchsize], 0)
      if loss_type == 'cos':
          norm1 = tf.sqrt(tf.reduce_sum(tf.multiply(fea1, fea1)))
          norm2 = tf.sqrt(tf.reduce_sum(tf.multiply(fea2, fea2)))
          return tf.reduce_sum(tf.multiply(fea1, fea2)) / norm1 / norm2

  def _conv_layer(self, x, in_filter, out_filter, stride, kernel_size, name):
      with tf.variable_scope(name):
          x = self._conv('conv', x, kernel_size, in_filter, out_filter, strides=stride)
          x = self._batch_norm('bn', x)
          x = self._relu(x, 0)
          return x


  def _temp_reduce_dim(self, x, in_dim, out_dim, name):
      with tf.variable_scope(name):
          x = self._fully_connected(x, out_dim, name='fc', in_dim=in_dim)
          x = self._batch_norm('bn', x)
          x = self._relu(x, 0)
          return x

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

  def id_residual(self, x, in_filter, hidden_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 1, in_filter, hidden_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0)
      x = self._conv('conv2', x, 3, hidden_filter, out_filter, [1, 1, 1, 1])

    # with tf.variable_scope('sub3'):
    #   x = self._batch_norm('bn3', x)
    #   x = self._relu(x, 0)
    #   x = self._conv('conv3', x, 1, hidden_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def conv_residual(self, x, in_filter, hidden_filter, out_filter, stride,
                  activate_before_residual=False):
      """Residual unit with 2 sub layers."""
      if activate_before_residual:
          with tf.variable_scope('shared_activation'):
              x = self._batch_norm('init_bn', x)
              x = self._relu(x, 0)
              orig_x = x
      else:
          with tf.variable_scope('residual_only_activation'):
              orig_x = x
              x = self._batch_norm('init_bn', x)
              x = self._relu(x, 0)

      with tf.variable_scope('sub_add'):
          orig_x = self._conv('conv1s', orig_x, 1, in_filter, out_filter, stride)

      with tf.variable_scope('sub1'):
          x = self._conv('conv1', x, 3, in_filter, hidden_filter, stride)

      with tf.variable_scope('sub2'):
          x = self._batch_norm('bn2', x)
          x = self._relu(x, 0)
          x = self._conv('conv2', x, 3, hidden_filter, out_filter, [1, 1, 1, 1])
          x += orig_x

      # with tf.variable_scope('sub3'):
      #     x = self._batch_norm('bn3', x)
      #     x = self._relu(x, 0)
      #     x = self._conv('conv3', x, 1, hidden_filter, out_filter, [1, 1, 1, 1])
      #



      tf.logging.debug('image after unit %s', x.get_shape())
      return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          self.precision, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n), dtype=self.precision))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim, name, in_dim=-1):
    """FullyConnected layer for final output."""
    with tf.variable_scope(name):
        prod_non_batch_dimensions=1
        if in_dim == -1:
            num_non_batch_dimensions = len(x.shape)
            prod_non_batch_dimensions = 1
            for ii in range(num_non_batch_dimensions - 1):
              prod_non_batch_dimensions *= int(x.shape[ii + 1])

        else:
            prod_non_batch_dimensions = in_dim
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim], dtype=self.precision,
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0, dtype=self.precision))
        b = tf.get_variable('biases', [out_dim], dtype=self.precision,
                            initializer=tf.constant_initializer(dtype=self.precision))
        return tf.nn.xw_plus_b(x, w, b)


  def _fully_connected_final(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.initializers.variance_scaling(distribution='uniform', dtype=self.precision))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer(dtype=self.precision))
    return tf.nn.xw_plus_b(x, w, b)

  def _reshape_cal_len(self, x):
      num_non_batch_dimensions = len(x.shape)
      prod_non_batch_dimensions = 1
      for ii in range(num_non_batch_dimensions - 1):
          prod_non_batch_dimensions *= int(x.shape[ii + 1])
      x = tf.reshape(x, [tf.shape(x)[0], -1])
      return x, prod_non_batch_dimensions

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])


  def _ave_pool(selfself, x, pool_size, strides):
    return tf.layers.average_pooling2d(x, pool_size, strides)
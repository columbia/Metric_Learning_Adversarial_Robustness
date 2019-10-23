"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.

Madry, A. et al. Towards deep learning models resistant to adversarial attacks. 2018.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from learning.model_vanilla import ModelVani
from utils import match_loss, triplet_loss_adversarial
from time import time
# from utils_folder.save_drebin import project_to_manifest

import dataloader.cifar10_input


class TargetedGen:
    def __init__(self, model_VarList, random_start, loss_type, dataset_type, momentum=0):
        self.rand = random_start
        self.momentum = momentum
        self.dataset_type = dataset_type
        self.loss_type = loss_type

        input_shape = None
        fea_len = 1024
        if dataset_type == "cifar10":
            input_shape = [None, 32, 32, 3]
            fea_len = 640

        self.weighted_lam = tf.placeholder(tf.float32, shape=None)
        self.x_B_origin = tf.placeholder(tf.float32, shape=input_shape)
        self.n_A_presoftmax_fixed = tf.placeholder(tf.float32, shape=[None, 10])
        self.x_Anat, self.y_A, self.x_Bnat, self.n_A4, self.n_A_presoftmax, self.n_B4, self.n_B_presoftmax, \
        self.is_training = model_VarList
        loss = None
        if loss_type == 'add_xent':
            # minimize the dist between Blogit and A to construct a targeted attack
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.n_B_presoftmax, labels=self.y_A)

            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin) ** 2) \
                   + (1 - self.weighted_lam) * (tf.reduce_mean((self.n_A_presoftmax_fixed - self.n_B_presoftmax) ** 2)
                   + 0.1 * tf.reduce_mean(y_xent))
        elif loss_type == 'Fea_mse':
            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin)**2) \
                    + (1 - self.weighted_lam) * tf.reduce_mean((self.n_A_presoftmax_fixed - self.n_B_presoftmax)**2)

        elif loss_type == 'n4_Fea_mse':
            self.n_A4_fixed = tf.placeholder(tf.float32, shape=[None, fea_len])
            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin)**2) \
                    + (1 - self.weighted_lam) * tf.reduce_mean((self.n_A4_fixed - self.n_B4)**2)

        elif loss_type == 'n4_add_xent':
            self.n_A4_fixed = tf.placeholder(tf.float32, shape=[None, fea_len])
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.n_B_presoftmax, labels=self.y_A)
            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin)**2) \
                    + (1 - self.weighted_lam) * (tf.reduce_mean((self.n_A4_fixed - self.n_B4)**2)
                                                 + 0.1 * tf.reduce_mean(y_xent))

        elif loss_type == 'xent_only':
            y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.n_B_presoftmax, labels=self.y_A)
            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin) ** 2) \
                   + (1 - self.weighted_lam) * (tf.reduce_mean(y_xent))

        elif loss_type == 'JSdiverge':
            episilon = 1e-8
            A_softmax = tf.nn.softmax(self.n_A_presoftmax_fixed)
            B_softmax = tf.nn.softmax(self.n_B_presoftmax)
            JS_loss = tf.reduce_mean(tf.reduce_sum(A_softmax * tf.log(A_softmax/(B_softmax + episilon))
                                                   + B_softmax * tf.log(B_softmax/(A_softmax + episilon)), axis=1),
                                     axis=0)

            loss = self.weighted_lam * tf.reduce_mean((self.x_Bnat - self.x_B_origin) ** 2) \
                   + (1 - self.weighted_lam) * JS_loss

        self.grad = tf.gradients(loss, self.x_Bnat)[0]

    def perturb(self, x, target_class_x, target_class_y, num_steps, step_size, lambda_value, epsilon, is_train, sess):
        x_raw = np.copy(x)
        if self.rand:
            x_init = x + np.random.uniform(-epsilon, epsilon, x.shape)
            x_init = np.clip(x_init, 0, 255)  # ensure valid pixel range
        else:
            x_init = np.copy(x)
        x = np.copy(x_init)
        if 'n4' in self.loss_type:
            np_n_A4_value = sess.run(self.n_A4, feed_dict={self.x_Anat: target_class_x,
                                                                                 self.is_training: is_train})
            grad = None
            for i in range(num_steps):
              if i == 0:
                grad = sess.run(self.grad, feed_dict={self.n_A4_fixed: np_n_A4_value,
                                                      self.x_B_origin: x_raw,
                                                      self.x_Bnat: x,
                                                      self.weighted_lam: lambda_value,
                                                      self.is_training: is_train,
                                                      self.y_A: target_class_y})
              else:
                grad_this = sess.run(self.grad, feed_dict={self.n_A4_fixed: np_n_A4_value,
                                                        self.x_B_origin: x_raw,
                                                        self.x_Bnat: x,
                                                        self.weighted_lam: lambda_value,
                                                        self.is_training: is_train,
                                                        self.y_A: target_class_y})
                grad = self.momentum * grad + (1 - self.momentum) * grad_this

              x = np.add(x, -step_size * np.sign(grad), out=x, casting='unsafe') #TODO: we are gradient decent, so minus step size
              if self.dataset_type == 'cifar10':
                x = np.clip(x, 0, 255)
              elif self.dataset_type == 'mnist':
                  x = np.clip(x, 0, 1)
        else:
            np_n_A_presoftmax_value = sess.run(self.n_A_presoftmax, feed_dict={self.x_Anat: target_class_x,
                                                                               self.is_training: is_train})
            grad = None
            for i in range(num_steps):
                if i == 0:
                    grad = sess.run(self.grad, feed_dict={self.n_A_presoftmax_fixed: np_n_A_presoftmax_value,
                                                          self.x_B_origin: x_raw,
                                                          self.x_Bnat: x,
                                                          self.weighted_lam: lambda_value,
                                                          self.is_training: is_train,
                                                          self.y_A: target_class_y})
                else:
                    grad_this = sess.run(self.grad, feed_dict={self.n_A_presoftmax_fixed: np_n_A_presoftmax_value,
                                                               self.x_B_origin: x_raw,
                                                               self.x_Bnat: x,
                                                               self.weighted_lam: lambda_value,
                                                               self.is_training: is_train,
                                                               self.y_A: target_class_y})
                    grad = self.momentum * grad + (1 - self.momentum) * grad_this

                x = np.add(x, -step_size * np.sign(grad), out=x,
                           casting='unsafe')  # TODO: we are gradient decent, so minus step size
                if self.dataset_type == 'cifar10':
                    x = np.clip(x, 0, 255)
                elif self.dataset_type == 'mnist':
                    x = np.clip(x, 0, 1)

        return x


class LinfPGDAttack:
  def __init__(self, model_VarList, epsilon, num_steps, step_size, random_start, loss_func, dataset_type,
               attack_suc_ratio=0, max_multi=1, use_momentum=False, momentum=0, pre_softmax=None, num_of_classes=10):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.epsilon = float(epsilon)
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.use_momentum = use_momentum
    self.momentum = momentum
    self.dataset_type = dataset_type

    self.attack_suc_ratio = attack_suc_ratio
    self.max_multi = max_multi

    self.a_pre_softmax = pre_softmax

    print('rand', self.rand, 'epsilon', epsilon, 'stepo', num_steps, 'step size', step_size, 'momentum', momentum)

    # self.n_xent, self.n_num_correct, self.n_accuracy, self.a_xent, \
    # self.a_num_correct, self.a_accuracy, self.disMatch_loss, \
    # self.x_nat, self.x_adv, self.y_input, self.is_training, self.a_pre_softmax, self.n_pre_softmax,\
    # self.x_neg, self.neg_pre_softmax = model_VarList
    self.x_adv, self.a_xent, self.y_input, self.is_training, self.a_Aaccuracy = model_VarList

    self.loss_func = loss_func

    if loss_func == 'xent':
      loss = self.a_xent
    elif loss_func == 'cw':
      assert pre_softmax !=None
      label_mask = tf.one_hot(self.y_input,
                              num_of_classes,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * self.a_pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * self.a_pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = self.a_xent

    self.grad = tf.gradients(loss, self.x_adv)[0]

    if self.dataset_type == 'drebin':
      self.sensitive_mask = np.load('Drebin_data/sensitive_mask.npy')[np.newaxis, :]

  def project_to_manifest(self, grad):
    return grad * self.sensitive_mask


  def perturb(self, x_nat, y, is_train, sess, make_int=False):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.dataset_type != 'drebin':
        x_nat = x_nat.astype(np.float32)
        x_init = np.copy(x_nat)
        if self.rand:
          if self.dataset_type == 'drebin':
              x_init = x_init.astype('float64') + self.project_to_manifest(np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape))
          else:
              x_init = x_init.astype('float64') + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
              # print('rand', self.rand, 'epsilon', self.epsilon, 'stepo', self.num_steps, 'step size', self.step_size, 'momentum',
              #       self.momentum)

          if self.dataset_type == 'drebin':
              x_init = np.clip(x_init, 0, 1)
          elif self.dataset_type == 'mnist'or self.dataset_type == 'imagenet_01':
              # print('image net epsilon', self.epsilon)
              x_init = np.clip(x_init, 0, 1)
          elif self.dataset_type == 'cifar10' or self.dataset_type == 'imagenet':
              x_init = np.clip(x_init, 0, 255)
          else:
              raise

        # print('range after fist random start', np.min(x_init), np.max(x_init))

        x = x_init
        cnt = 1
        while cnt <= self.max_multi:
            for i in range(self.num_steps):
              if i == 0:
                  grad = sess.run(self.grad, feed_dict={self.x_adv: x,
                                                      self.y_input: y,
                                                      self.is_training: is_train})
              else:
                  grad_this = sess.run(self.grad, feed_dict={self.x_adv: x,
                                                           self.y_input: y,
                                                           self.is_training: is_train})
                  grad = self.momentum * grad + (1 - self.momentum) * grad_this

              grad_sign = np.sign(grad)


              if self.dataset_type == 'drebin':
                  grad_sign = self.project_to_manifest(grad_sign)

              x = np.add(x, self.step_size * grad_sign, out=x, casting='unsafe')

              #TODO:gradient ascent
              # print('epsilon', self.epsilon)
              # print('type', x_nat.dtype, x.dtype)
              # print('x_nat - self.epsilon', np.min(x_nat - self.epsilon), np.max(x_nat - self.epsilon))
              # print('x_nat + self.epsilon', np.min(x_nat + self.epsilon), np.max(x_nat + self.epsilon))
              assert x_nat.dtype != 'uint8'  # TODO: it will result in serious error, because 255+8 = 7, when add epsilon, it is catostraphy
              # assert x_nat.dtype == 'float32'

              # print(x_nat.dtype, self.epsilon, type(self.epsilon))
              x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)

              # temp = x - x_nat
              # temp = temp.astype(int)
              # print('diff', np.min(temp), np.max(temp))


              if self.dataset_type == 'cifar10' or self.dataset_type == 'imagenet':
                  x = np.clip(x, 0, 255)
              elif self.dataset_type == 'mnist' or self.dataset_type == 'imagenet_01':
                  x = np.clip(x, 0, 1)
              elif self.dataset_type == 'drebin':
                  x = np.clip(x, 0, 1)

            # accuracy = sess.run(self.a_Aaccuracy, feed_dict={self.x_adv: x,
            #                                           self.y_input: y,
            #                                           self.is_training: is_train})
            # if accuracy<(1 - self.attack_suc_ratio):
            #     break
            cnt += 1

        if self.dataset_type == 'drebin':
            x = np.rint(x)
        if not is_train and self.dataset_type in ['cifar10', 'imagenet']:
            x = np.rint(x)
        elif make_int:
            x = np.rint(x)


        # if self.dataset_type == 'drebin':
        #     print('-'*10, 'Perturbation', '-'*10)
        #     print(t22-t21)
        #     print(t23-t22)
        #     print(t24-t23)
        #     print(t25-t24)
        #     print(t26-t25)
        #     print(t27-t26)
        #     print()
        #     print(t1-t0)
        #     print(t2-t1)
        #     print(t3-t2)
    else:
        # batch_size = x_nat.shape[0]
        # x = np.copy(x_nat)
        #
        #
        #
        # x_loop = tf.placeholder(tf.float32, shape=[batch_size, 545334])
        # y_loop = tf.placeholder(tf.int64, shape=[batch_size])
        # is_train_loop = tf.placeholder(tf.bool, shape=[])
        # count = tf.constant(0)
        # sensitive_mask = tf.convert_to_tensor(self.sensitive_mask, dtype=tf.float32)
        # model = ModelDrebin()
        #
        # def condition(x_loop, y_loop, is_train_loop, count):
        #     return count < self.num_steps
        #
        # def body(x_loop, y_loop, is_train_loop, count):
        #     zero_mask =  tf.cast(tf.equal(x_loop, 0), tf.float32)
        #
        #     returned = model._encoder(x_loop, y_loop, is_train_loop, mask_effective_attack=None)
        #     a_Axent = returned[1]
        #     grad = tf.gradients(a_Axent, x_loop)[0]
        #     grad *= zero_mask * sensitive_mask
        #
        #     grad_max = tf.reduce_max(grad, reduction_indices=[1])
        #     grad_max_mask = tf.cast(tf.equal(grad, tf.expand_dims(grad_max, axis=1)), tf.float32)
        #     x_loop += tf.ones_like(grad) * grad_max_mask
        #
        #     count += 1
        #
        #     return x_loop, y_loop, is_train_loop, count
        #
        # result_x, _, _, _ = tf.while_loop(condition, body, (x_loop, y_loop, is_train_loop, count), back_prop=False)
        #
        # output_x = sess.run(result_x, feed_dict={ x_loop: x,
        #                                           y_loop: y,
        #                                           is_train_loop: is_train})

        x = np.copy(x_nat)
        for i in range(self.num_steps):
            zero_mask = x == 0
            grad = sess.run(self.grad, feed_dict={self.x_adv: x,
                                                  self.y_input: y,
                                                  self.is_training: is_train})
            grad = self.project_to_manifest(grad * zero_mask)
            max_inds = np.argmax(grad, axis=1)
            x[range(x.shape[0]), max_inds] = 1
    output_x = x

    return output_x


if __name__ == '__main__':
  import json
  import sys
  import math

  from utils import trainable_in


  from model import Model
  from model_adv import ModelAdv

  with open('config.json') as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir_pgd_load'])
  print('model file', model_file)
  if model_file is None:
    print('No model found')
    sys.exit()

  # TODO: change this to specified model
  # model = Model(mode='eval')
#  model = ModelAdv(mode='eval')
  model = ModelVani()

  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'],
                         # config['use_momentum'],
                         # config['attack_momentum']
                         )
  saver = tf.train.Saver()

  var_main_encoder = trainable_in('main_encoder')
  saver_restore = tf.train.Saver(var_main_encoder)

  data_path = config['data_path']
  cifar = dataloader.cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    sess.run(tf.global_variables_initializer())
    model_file = tf.train.latest_checkpoint(config['model_dir_pgd_load'])
    saver_restore.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

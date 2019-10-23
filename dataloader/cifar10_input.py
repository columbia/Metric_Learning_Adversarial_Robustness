"""
Utilities for importing the CIFAR10 dataset.

Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
version = sys.version_info

import numpy as np
import matplotlib.pyplot as plt



class CIFAR10Data(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).

    Inputs to constructor
    =====================

        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.

    """
    def __init__(self, path):
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
            train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

    @staticmethod
    def _load_datafile(filename):
      with open(filename, 'rb') as fo:
          if version.major == 3:
              data_dict = pickle.load(fo, encoding='bytes')
          else:
              data_dict = pickle.load(fo)

          assert data_dict[b'data'].dtype == np.uint8
          image_data = data_dict[b'data']
          image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
          return image_data, np.array(data_dict[b'labels'])


class CIFAR100_Data(object):
    def __init__(self, path):
        train_images, train_labels = self._load_datafile(os.path.join(path, 'train'), 50000)
        eval_images, eval_labels = self._load_datafile(os.path.join(path, 'test'), 10000)

        self.train_data = DataSubset(train_images, train_labels)
        self.eval_data = DataSubset(eval_images, eval_labels)

        metadata_filename = 'meta'
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            self.label_names = data_dict[b'fine_label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')

    @staticmethod
    def _load_datafile(filename, examples_num):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)
            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((examples_num, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'fine_labels'])



class SepClaCIFAR_100(object):
    def __init__(self, path, fine=True):
        self.fine = fine
        try:
            train_images_classed = np.load(os.path.join(path, 'train_classed100.npy'))
            test_images_classed = np.load(os.path.join(path, 'test_classed100.npy'))
            # train_class_label = np.load(os.path.join(path, 'train_classed_label.npy'))
        except:
            train_images, train_labels = self._load_datafile(os.path.join(path, 'train'), 50000)
            eval_images, eval_labels = self._load_datafile(os.path.join(path, 'test'), 10000)

            train_images_classed = np.zeros((100, 500, 32, 32, 3), dtype='uint8')
            cnt_class = np.zeros((100), dtype='int32')

            ############################
            for ii in range(50000):
                label_int = int(train_labels[ii])
                # print(cnt_class)
                train_images_classed[label_int, int(cnt_class[label_int]), ...] = train_images[ii, ...]
                cnt_class[label_int] += 1

            test_images_classed = np.zeros((100, 100, 32, 32, 3), dtype='uint8')
            test_cnt_class = np.zeros((100), dtype='int32')

            for ii in range(10000):
                label_int = int(eval_labels[ii])
                test_images_classed[label_int, int(test_cnt_class[label_int]), ...] = eval_images[ii, ...]
                test_cnt_class[label_int] += 1

            np.save(os.path.join(path, 'train_classed100.npy'), train_images_classed)
            np.save(os.path.join(path, 'test_classed100.npy'), test_images_classed)

        self.train_data = DataClasSubset(train_images_classed, 100)
        self.eval_data = DataClasSubset(test_images_classed, 100)

        metadata_filename = 'meta'
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'fine_label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')
        print("label names", self.label_names)

    def _load_datafile(self, filename, examples_num):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)
            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((examples_num, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'fine_labels']) if self.fine else np.array(data_dict[b'coarse_labels'])



class SepClaCIFAR10(object):
    def __init__(self, path):
        try:
            train_images_classed = np.load(os.path.join(path, 'train_classed.npy'))
            test_images_classed = np.load(os.path.join(path, 'test_classed.npy'))
            # train_class_label = np.load(os.path.join(path, 'train_classed_label.npy'))
        except:
            train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
            eval_filename = 'test_batch'


            train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
            train_labels = np.zeros(50000, dtype='int32')
            for ii, fname in enumerate(train_filenames):
                cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
                train_images[ii * 10000 : (ii+1) * 10000, ...] = cur_images
                train_labels[ii * 10000 : (ii+1) * 10000, ...] = cur_labels

            train_images_classed = np.zeros((10, 5000, 32, 32, 3), dtype='uint8')
            cnt_class = np.zeros((10), dtype='int32')

            ############################
            for ii in range(50000):
                label_int = int(train_labels[ii])
                train_images_classed[label_int, int(cnt_class[label_int]), ...] = train_images[ii, ...]
                cnt_class[label_int] += 1

            eval_images, eval_labels = self._load_datafile(
                os.path.join(path, eval_filename))
            test_images_classed = np.zeros((10, 1000, 32, 32, 3), dtype='uint8')
            test_cnt_class = np.zeros((10), dtype='int32')

            for ii in range(10000):
                label_int = int(eval_labels[ii])
                test_images_classed[label_int, int(test_cnt_class[label_int]), ...] = eval_images[ii, ...]
                test_cnt_class[label_int] += 1

            np.save(os.path.join(path, 'train_classed.npy'), train_images_classed)
            np.save(os.path.join(path, 'test_classed.npy'), test_images_classed)

        self.train_data = DataClasSubset(train_images_classed)
        self.eval_data = DataClasSubset(test_images_classed)

        metadata_filename = 'batches.meta'
        with open(os.path.join(path, metadata_filename), 'rb') as fo:
              if version.major == 3:
                  data_dict = pickle.load(fo, encoding='bytes')
              else:
                  data_dict = pickle.load(fo)

              self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')
        print("label names", self.label_names)

    def _load_datafile(self, filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])


class DataClasSubset(object):
    def __init__(self, xs, num_of_class=10):
        self.xs = xs
        self.n = xs.shape[1]
        self.data_start = np.zeros((num_of_class), dtype='int32')
        self.num_of_class = num_of_class
        self.cur_order = np.random.permutation(self.n)

    def get_next_data_basedon_class(self, target_class, reshuffle_after_pass=True):
        batch_size = target_class.shape[0]
        batch_out = np.zeros((batch_size, 32, 32, 3), dtype='uint8')

        for ii in range(batch_size):
            label_int = int(target_class[ii])
            batch_out[ii, ...] = self.xs[label_int, self.cur_order[self.data_start[label_int]]]
            self.data_start[label_int] += 1

            if self.data_start[label_int] >= self.xs.shape[1]:
                if reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.n)
                self.data_start = np.zeros((self.num_of_class), dtype='int32')

        return batch_out, target_class



class AugmentedCIFAR10Data(object):
    """
    Data augmentation wrapper over a loaded dataset.

    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """
    def __init__(self, raw_cifar10data, sess, model):
        # assert isinstance(raw_cifar10data, CIFAR10Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_cifar10data.train_data, sess,
                                             self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_cifar10data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_cifar10data.label_names


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        # images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]

    def get_next_data_basedon_class(self, y_batch, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_data_basedon_class(y_batch, reshuffle_after_pass)
        # images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                    raw_batch[0]}), raw_batch[1]

if __name__ == '__main__':
    import json
    # from model_vanilla import ModelVani

    # gpu_options = tf.GPUOptions(allow_growth=True)
    with open('config_cifar100.json') as config_file:
        config = json.load(config_file)

    data_path = config['data_path']
    # raw_cifar = CIFAR10Data(data_path)
    # cla_raw_cifar = SepClaCIFAR10(data_path)
    raw_cifar = CIFAR100_Data(data_path)
    cla_raw_cifar = SepClaCIFAR_100(data_path)

    # model = ModelVani()
    batch_size = 5

    # a=np.load('./cifar10_data/train_classed.npy')
    #
    # cnt = 0
    # for ii in range(5000):
    #     if np.max(a[1, ii]) < 1:
    #         cnt += 1
    # print('cnt', cnt)

    with tf.Session(config=tf.ConfigProto()) as sess:
        cifar = AugmentedCIFAR10Data(raw_cifar, sess, '')
        cifar_aux = AugmentedCIFAR10Data(cla_raw_cifar, sess, '')

        # model_dir_load = tf.train.latest_checkpoint(config['model_load_dir'])
        # saver_restore.restore(sess, model_dir_load)

        for i in range(10):

            x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                                   multiple_passes=True)
            print(type(y_batch), y_batch.shape)
            x_batch_2, y_batch_2 = cifar_aux.train_data.get_next_data_basedon_class(y_batch)
            print(type(x_batch_2), x_batch_2.shape)
            print(type(y_batch_2), y_batch_2.shape)

            print(y_batch[0])
            plt.imshow(x_batch_2[0]/255)
            plt.show()

'''
Script to save mnist locally
'''
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

if not os.path.isdir('../MNIST_data'):
    os.mkdir('../MNIST_data')

    raw_dataset = input_data.read_data_sets('../MNIST_data', one_hot=False)
    train_data = raw_dataset.train.images, raw_dataset.train.labels
    test_data = raw_dataset.test.images, raw_dataset.test.labels

    np.savez('../MNIST_data/train', images=raw_dataset.train.images, labels=raw_dataset.train.labels)
    np.savez('../MNIST_data/test', images=raw_dataset.test.images, labels=raw_dataset.test.labels)


test = np.load('../MNIST_data/test.npz')
print(test['images'].shape, test['labels'].shape)

import tensorflow as tf
from dataloader.datasets import dataset_factory
import matplotlib.pyplot as plt

class DataSubset(object):
    def __init__(self, mode, is_training, batch_size, dataset_location, sess):
        img_bounds = (0, 1)
        self.mode = mode
        self.is_training = is_training
        self.batch_size = batch_size
        self.dataset, examples_per_epoch, num_classes, bounds = (
          dataset_factory.get_dataset(
            'tiny_imagenet',
            self.mode,
            self.batch_size,
            64,
            is_training=self.is_training,
            bounds=img_bounds,
            dataset_location=dataset_location))
        self.sess = sess


    def get_next_batch(self, batch, multiple_passes=True):
        dataset_iterator = self.dataset.make_one_shot_iterator()
        images, labels = dataset_iterator.get_next()

        return self.sess.run([images, labels])



class AugmentedIMAGENETData(object):
    def __init__(self, batch_size, dataset_location, sess):
        self.train_data = DataSubset('train', True, batch_size, dataset_location, sess)
        self.eval_data = DataSubset('validation', False, 10000, dataset_location, sess)




if __name__ == "__main__":
    batch_size = 5
    dataset_location = 'tiny-imagenet-tfrecord'



    with tf.Session(config=tf.ConfigProto()) as sess:
        cifar = AugmentedIMAGENETData(batch_size, dataset_location, sess)
        xs, ys = cifar.train_data.get_next_batch()

        plt.imshow(xs[0])
        plt.show()

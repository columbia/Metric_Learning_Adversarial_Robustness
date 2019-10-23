import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from timeit import default_timer as timer

def tiny_imagenet_parser(value, image_size, is_training, bounds):
  """Parses tiny imagenet example.

  Args:
    value: an image.
    image_size: size of the image.
    is_training: if True then do training preprocessing (which includes
      random cropping), otherwise do eval preprocessing.

  Returns:
    image: tensor with the image.
    label: true label of the image.
  """
  image = value

  # Crop image
  if is_training:
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
                                   dtype=tf.float32,
                                   shape=[1, 1, 4]),
        min_object_covered=0.5,
        aspect_ratio_range=[0.75, 1.33],
        area_range=[0.5, 1.0],
        max_attempts=20,
        use_image_if_no_bounding_boxes=True)
    image = tf.slice(image, bbox_begin, bbox_size)
    # Data augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, 0.5, 2.0)

  # resize image
  image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

  # Rescale image to [-1, 1] range.
  if bounds == (-1, 1):
      image = tf.multiply(tf.subtract(image, 0.5), 2.0)

  image = tf.reshape(image, [image_size, image_size, 3])
  return image


class AugmentedDataSubset(object):
    def __init__(self, raw_dataset, sess, is_training):
        self.raw_dataset = raw_dataset
        self.sess = sess
        self.is_training = is_training


    def get_next_batch(self, batch_size, multiple_passes=True):
        t0 = timer()
        image_size = 64
        bounds = (0, 1)
        raw_x_batch, raw_y_batch  = self.raw_dataset.get_next_batch(batch_size)
        t1 = timer()
        dataset = tf.data.Dataset.from_tensor_slices(raw_x_batch)
        dataset = dataset.repeat()
        t2 = timer()
        dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: tiny_imagenet_parser(value, image_size, self.is_training, bounds),
            batch_size=batch_size,
            num_parallel_batches=4,
            drop_remainder=True))
        iter = dataset.make_one_shot_iterator()
        processed_x_batch = iter.get_next()
        xs, ys = self.sess.run(processed_x_batch), raw_y_batch
        t3 = timer()

        print('-'*10)
        print(t1-t0)
        print(t2-t1)
        print(t3-t2)

        return xs, ys

    def get_next_data_basedon_class(self, target_class, reshuffle_after_pass=True):
        image_size = 64
        bounds = (0, 1)
        raw_x_batch, raw_y_batch  = self.raw_dataset.get_next_data_basedon_class(target_class, reshuffle_after_pass)

        batch_size = target_class.shape[0]
        dataset = tf.data.Dataset.from_tensor_slices(raw_x_batch)
        dataset = dataset.repeat()

        dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            lambda value: tiny_imagenet_parser(value, image_size, self.is_training, bounds),
            batch_size=batch_size,
            num_parallel_batches=1,
            drop_remainder=True))

        iter = dataset.make_one_shot_iterator()
        processed_x_batch = iter.get_next()

        return self.sess.run(processed_x_batch), raw_y_batch



class AugmentedIMAGENETData(object):
    def __init__(self, raw_data, sess):

        self.train_data = AugmentedDataSubset(raw_data.train_data, sess, True)
        self.eval_data = AugmentedDataSubset(raw_data.eval_data, sess, False)


if __name__ == "__main__":

    from mnist_input import MNISTData, MNISTDataClassed

    raw = MNISTData('../imagenet_data', 'imagenet')
    cla_raw = MNISTDataClassed('../imagenet_data', 'imagenet')

    with open("../imagenet_data/label_set.pkl", "rb") as fp:
        label_set = pickle.load(fp)
    with open('../imagenet_data/str_to_class.pkl', 'rb') as fp:
        str_to_class = pickle.load(fp)

    with tf.Session(config=tf.ConfigProto()) as sess:
        cifar = AugmentedIMAGENETData(raw, sess)
        for i in range(3):
            xs, ys = cifar.train_data.get_next_batch(4)
            # print(str_to_class[label_set[ys[0]]])
            # plt.imshow(xs[0]/255)
            # plt.show()

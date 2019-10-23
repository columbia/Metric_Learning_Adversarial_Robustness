import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

# def tiny_imagenet_parser(image, image_size, is_training):
#     if is_training:
#         # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
#         #     tf.shape(image),
#         #     bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0],
#         #                                dtype=tf.float32,
#         #                                shape=[1, 1, 4]),
#         #     min_object_covered=0.5,
#         #     aspect_ratio_range=[0.75, 1.33],
#         #     area_range=[0.5, 1.0],
#         #     max_attempts=20,
#         #     use_image_if_no_bounding_boxes=True)
#         # image = tf.slice(image, bbox_begin, bbox_size)
#         # image = tf.image.random_flip_left_right(image)
#         # image = tf.image.random_saturation(image, 0.5, 2.0)
#
#         image = tf.image.resize_image_with_crop_or_pad(image, image_size + 8, image_size + 8)
#         image = tf.random_crop(image, [image_size, image_size, 3])
#         image = tf.image.random_flip_left_right(image)
#
#     # resize image
#     # image = tf.image.resize_bicubic([image], [image_size, image_size])[0]
#     # image = tf.reshape(image, [image_size, image_size, 3])
#
#     return image


class AugmentedDataSubset(object):
    def __init__(self, raw_dataset, sess, augmented, x_input_placeholder):
        self.raw_dataset = raw_dataset
        self.sess = sess
        self.augmented = augmented
        self.x_input_placeholder = x_input_placeholder

    def get_next_batch(self, batch_size, multiple_passes=True):
        raw_x_batch, raw_y_batch  = self.raw_dataset.get_next_batch(batch_size)

        xs, ys = self.sess.run(self.augmented, feed_dict={self.x_input_placeholder: raw_x_batch}), raw_y_batch

        return xs, ys

    def get_next_data_basedon_class(self, target_class, reshuffle_after_pass=True):
        raw_x_batch, raw_y_batch  = self.raw_dataset.get_next_data_basedon_class(target_class, reshuffle_after_pass)

        xs, ys = self.sess.run(self.augmented, feed_dict={self.x_input_placeholder: raw_x_batch}), raw_y_batch

        return xs, ys


class AugmentedIMAGENETData(object):
    def __init__(self, raw_data, sess):
        self.image_size = 64
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, self.image_size, self.image_size, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 8, self.image_size + 8),
            self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size, self.image_size, 3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped


        self.train_data = AugmentedDataSubset(raw_data.train_data, sess, self.augmented, self.x_input_placeholder)
        # self.eval_data = AugmentedDataSubset(raw_data.eval_data, sess, self.augmented, self.x_input_placeholder)


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

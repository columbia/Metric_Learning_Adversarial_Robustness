"""
Utilities for importing the MNIST/Drebin/Tiny ImageNet dataset.

Each image in the dataset is a numpy array of shape (28, 28, 1), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""
import os
import sys
import json
import pickle
import time
import tensorflow as tf
import numpy as np
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt

sys.path.insert(0,'../')
from utils_folder.save_drebin import save_sparse_csr, load_sparse_csr, show_idx, get_non_zero_indices
from learning.model_mnist import ModelMNIST
#from model_imagenet import ModelIMAGENET



drebin_num_of_features = 545334

# Convert dictionary to an object
class LoadData:
    def __init__(self, filename, data_size=None):
        '''
        data_size: the number of training datapoints to use. If None is given, all data points are used.
        '''
        data = load_sparse_csr(filename)
        if data_size == None:
            data_size = data['images'].shape[0]
        data_size = np.min([data_size, data['images'].shape[0]])

        # Set seed in order to make sure the chosen data to be the same for regular dataset and classed dataset
        np.random.seed(0)
        cur_order = np.random.permutation(data['images'].shape[0])
        self.images = data['images'][cur_order[:], ...][:data_size]
        self.labels = data['labels'][cur_order[:], ...][:data_size]
        self.data_size = data_size




class MNISTData:
    '''
    train_size: the number of training datapoints to use. If None is given, all data points are used.
    test_size: the number of testing datapoints to use. If None is given, all data points are used.
    '''
    def __init__(self, folder_path, dataset, train_size=None, test_size=None):
        train_path = os.path.join(folder_path, 'train.npz')
        test_path = os.path.join(folder_path, 'test.npz')

        train = LoadData(train_path, train_size)
        test = LoadData(test_path, test_size)

        train_size = train.data_size
        test_size = test.data_size



        if dataset == 'mnist':
            self.train_data = DataClasSubset(train.images.reshape(train_size, 28, 28), train.labels, dataset)
            self.eval_data = DataClasSubset(test.images.reshape(test_size, 28, 28), test.labels, dataset)
        elif dataset == 'drebin':
            self.train_data = DataClasSubset(train.images, train.labels, dataset)
            self.eval_data = DataClasSubset(test.images, test.labels, dataset)
        elif dataset == 'imagenet':
            # train_mean = np.mean(train.images, axis=0)
            # train_std = np.std(train.images, axis=0)
            #
            # self.train_data = DataClasSubset((train.images-train_mean)/train_std, train.labels, dataset)
            # self.eval_data = DataClasSubset((test.images-train_mean)/train_std, test.labels, dataset)

            self.train_data = DataClasSubset(train.images, train.labels, dataset)
            self.eval_data = DataClasSubset(test.images, test.labels, dataset)

        # if dataset == 'imagenet':
        #     with open("../imagenet_data/label_set.pkl", "rb") as fp:
        #         label_set = pickle.load(fp)
        #     with open('../imagenet_data/str_to_class.pkl', 'rb') as fp:
        #         str_to_class = pickle.load(fp)
        #     print('init', str_to_class[label_set[train.labels[0]]])
        #     print('init', str_to_class[label_set[test.labels[0]]])
        #
        #     plt.imshow(train.images[0])
        #     plt.show()
        #     plt.imshow(test.images[0])
        #     plt.show()



class MNISTDataClassed:
    '''
    train_size: the number of training datapoints to use. If None is given, all data points are used.
    test_size: the number of testing datapoints to use. If None is given, all data points are used.
    reprocess: If reprocess the data (i.e. group data) by loading from the source data file. This is default to false to save preprocessing time. However, it is necessary when only a fraction of the original data in train.npz and test.npz is used.
    '''
    def __init__(self, folder_path, dataset, train_size=None, test_size=None, reprocess=False):
        self.dataset = dataset
        if dataset == 'mnist':
            self.class_num = 10
        elif dataset == 'drebin':
            self.class_num = 2
        elif dataset == 'imagenet':
            self.class_num = 200

        train_path = os.path.join(folder_path, 'train.npz')
        test_path = os.path.join(folder_path, 'test.npz')
        train_classed_path = os.path.join(folder_path, 'train_classed.npz')
        test_classed_path = os.path.join(folder_path, 'test_classed.npz')

        self.train_data = self.try_load(train_path, train_classed_path, train_size, reprocess)
        self.eval_data = self.try_load(test_path, test_classed_path, test_size, reprocess)

    def try_load(self, src_path, end_path, data_size, reprocess):
        # If data_size != None, i.e. only a portion of data is used,
        if os.path.exists(end_path) and (data_size == None or reprocess == False):
            npz_file = load_sparse_csr(end_path)
            images_classed = [npz_file['arr_'+str(i)] for i in range(self.class_num)]
            num_in_each_class = npz_file['arr_'+str(self.class_num)]
        else:
            images_and_labels = LoadData(src_path, data_size)
            images_classed, num_in_each_class = self.group_imgs(images_and_labels)

            images_classed.append(num_in_each_class)
            save_sparse_csr(end_path,**{'arr_'+str(i):ic for i, ic in enumerate(images_classed)})
        # print(images_classed[0].shape, num_in_each_class)
        return DataSubsetClassed(images_classed, num_in_each_class, self.dataset)

    def group_imgs(self, data):
        if self.dataset == 'mnist':
            images = data.images.reshape([data.images.shape[0], 28, 28])
        elif self.dataset == 'drebin':
            images = data.images
        elif self.dataset == 'imagenet':
            images = data.images


        labels = data.labels

        num_in_each_class = [0 for _ in range(self.class_num)]
        start_in_each_class = [0 for _ in range(self.class_num)]
        images_classed = []

        # Count the number of images in each class
        for label in labels:
            num_in_each_class[int(label)] += 1

        for i in range(self.class_num):
            if self.dataset == 'mnist':
                images_classed.append(np.zeros((num_in_each_class[i], 28, 28)))
            elif self.dataset == 'drebin':
                images_classed.append(lil_matrix((num_in_each_class[i], drebin_num_of_features), dtype=np.int8))
            elif self.dataset == 'imagenet':
                images_classed.append(np.zeros((num_in_each_class[i], 64, 64, 3), dtype=np.uint8))
        s = time.time()
        for ii in range(images.shape[0]):
            if self.dataset == 'drebin' and ii % 50 == 0:
                print(ii, '/', images.shape[0], time.time()-s)
            label_int = int(labels[ii])
            images_classed[label_int][start_in_each_class[label_int]] = images[ii]
            start_in_each_class[label_int] += 1

        # plt.imshow(images_classed[0][0])
        # plt.show()

        return images_classed, num_in_each_class

class DataClasSubset:
    def __init__(self, xs, ys, dataset):
        '''
        INPUT:
            xs: (num of data points in total, img_dims,...)
            ys: (num of data points in total,)
        '''
        self.xs = xs
        self.ys = ys
        self.dataset = dataset
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.xs.shape[0])
        self.xs = self.xs[self.cur_order[:], ...]
        self.ys = self.ys[self.cur_order[:], ...]

    def get_next_batch(self, batch_size, multiple_passes=True, reshuffle_after_pass=True):
        '''
        Provide a batch of (images, labels).
        If multiple_passes is False and we run out of data points, None, None will be returned.
        INPUT:
            batch_size
        OUTPUT:
            batch_xs, (batch_size, img_dims...)
            batch_ys, (batch_size,)
        '''
        if self.xs.shape[0] < batch_size:
            raise ValueError('Batch size can be at most the dataset size,'+str(batch_size)+' versus '+str(self.xs.shape[0]))

        actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)

        if actual_batch_size < batch_size:
            if multiple_passes:
                if reshuffle_after_pass:
                    self.cur_order = np.random.permutation(self.xs.shape[0])
                self.batch_start = 0
                actual_batch_size = min(batch_size, self.xs.shape[0] - self.batch_start)
            else:
                if actual_batch_size <= 0:
                    return None, None

        batch_end = self.batch_start + actual_batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size

        if self.dataset == 'drebin':
            batch_xs = batch_xs.toarray()
        return batch_xs, batch_ys


class DataSubsetClassed:
    def __init__(self, xs_classed, num_in_each_class, dataset):
        '''
        INPUT:
         xs_classed: list of numpy.ndarray, list of (num of data points in each class, img_dims,...)
        '''
        if dataset == 'mnist':
            self.class_num = 10
        elif dataset == 'drebin':
            self.class_num = 2
        elif dataset == 'imagenet':
            self.class_num = 200

        self.xs_classed = xs_classed
        self.num_in_each_class = num_in_each_class
        self.data_start = [0 for _ in range(self.class_num)]
        self.cur_order = [np.random.permutation(self.xs_classed[i].shape[0]) for i in range(self.class_num)]
        self.dataset = dataset



    def get_next_data_basedon_class(self, target_class, reshuffle_after_pass=True):
        '''
        INPUT:
         target_class: numpy.ndarray, (batch_size,)
        OUTPUT:
         batch_out: (batch_size, img_dim1, img_dim2, img_dim3)
         target_class: numpy.ndarray, (batch_size,)
        '''
        batch_size = target_class.shape[0]
        if self.dataset == 'mnist':
            batch_out = np.zeros((batch_size, 28, 28))
        elif self.dataset == 'drebin':
            batch_out = np.zeros((batch_size, drebin_num_of_features), dtype=np.uint8)
            # batch_out = lil_matrix((batch_size, drebin_num_of_features), dtype=np.uint8)
        elif self.dataset == 'imagenet':
            batch_out = np.zeros((batch_size, 64, 64, 3), dtype=np.uint8)

        for ii in range(batch_size):
            label_int = int(target_class[ii])
            # Take an image from the randomized dataset grouped by class label
            tmp = (self.xs_classed[label_int][self.cur_order[label_int][self.data_start[label_int]]])
            if self.dataset == 'drebin':
                tmp = tmp.toarray()
            batch_out[ii, ...] = tmp

            # print(label_int, self.cur_order[self.data_start[label_int]])

            self.data_start[label_int] += 1

            if self.data_start[label_int] >= self.xs_classed[label_int].shape[0]:
                if reshuffle_after_pass:
                    self.cur_order = [np.random.permutation(self.xs_classed[i].shape[0]) for i in range(self.class_num)]
                self.data_start = [0 for _ in range(self.class_num)]

        # if self.dataset == 'drebin':
        #     batch_out = batch_out.toarray()

        return batch_out, target_class


if __name__ == '__main__':
    dataset = 'imagenet'

    if dataset == 'drebin':
        # os.system('python save_drebin.py')
        model = ModelDrebin()
        config_path = '../config_drebin.json'
    elif dataset == 'mnist':
        # os.system('python save_mnist.py')
        model = ModelMNIST()
        config_path = '../config_mnist.json'
    elif dataset == 'imagenet':
        # os.system('python save_imagenet.py')
        #model = ModelIMAGENET()
        config_path = '../config_imagenet.json'


    # gpu_options = tf.GPUOptions(allow_growth=True)
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    data_path = config['data_path']

    raw = MNISTData('../'+data_path, dataset)
    cla_raw = MNISTDataClassed('../'+data_path, dataset)

    batch_size = 5


    benign_idx_list_list = []
    mal_idx_list_list = []
    if dataset == 'drebin':
        mal_filenames = []
        with open('Drebin_data/sha256_family.csv', 'r') as f_in:
            for i, line in enumerate(f_in):
                if i != 0:
                    filename = line.strip('\n').split()[0]
                    mal_filenames.append(filename)


        with open('Drebin_data/processed_apps_list.txt', 'r') as f_in:
            for line in f_in:
                cur_filename = line.strip('\n')
                idx_list = show_idx('Drebin_data/feature_vectors/'+cur_filename)
                if cur_filename in mal_filenames:
                    mal_idx_list_list.append(idx_list)
                else:
                    benign_idx_list_list.append(idx_list)

            print('Drebin label sums')
            train = load_sparse_csr('Drebin_data/train.npz')
            test = load_sparse_csr('Drebin_data/test.npz')
            print('train', np.sum(train['labels']))
            print('test', np.sum(test['labels']))
            print(list(train['labels'].nonzero()[0]))
            print('-'*30)


    with tf.Session(config=tf.ConfigProto()) as sess:
    # model_dir_load = tf.train.latest_checkpoint(config['model_load_dir'])
    # saver_restore.restore(sess, model_dir_load)
        for i in range(3):
            x_batch, y_batch = raw.eval_data.get_next_batch(batch_size)

            x_batch_2, y_batch_2 = cla_raw.eval_data.get_next_data_basedon_class(y_batch)

            print('-'*50)
            print(np.sum(x_batch[0]))
            print(y_batch[0])

            print('-'*50)
            print(np.sum(x_batch_2[0]))
            print(y_batch_2[0])

            if dataset == 'imagenet':
                with open("../imagenet_data/label_set.pkl", "rb") as fp:
                    label_set = pickle.load(fp)
                with open('../imagenet_data/str_to_class.pkl', 'rb') as fp:
                    str_to_class = pickle.load(fp)
                print(str_to_class[label_set[y_batch[0]]])
                print(str_to_class[label_set[y_batch_2[0]]])

            print(x_batch.shape, y_batch.shape, x_batch_2.shape, y_batch_2.shape)
            # print(x_batch[0][15, :])

            if dataset == 'drebin':

                sort_indices = get_non_zero_indices(x_batch_2[0])
                print(len(mal_idx_list_list), len(benign_idx_list_list))
                if sort_indices in mal_idx_list_list:
                    assert y_batch_2[0] == 1
                elif sort_indices in benign_idx_list_list:
                    assert y_batch_2[0] == 0
                else:
                    print('unfound', type(sort_indices), type(benign_idx_list_list[0]), sort_indices, benign_idx_list_list[0])

            else:
                plt.imshow(x_batch[0])
                plt.show()

                plt.imshow(x_batch_2[0])
                plt.show()

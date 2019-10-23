'''
Script to save tiny imagenet locally
'''
import numpy as np
import os
import re
import imageio
import pickle
from time import time


import matplotlib.pyplot as plt



if __name__ == '__main__':
    data_folder = '../imagenet_data'
    if not os.path.isdir(data_folder):
        os.mkdir(data_folder)

    str_to_class = dict()
    with open(os.path.join(data_folder, 'words.txt'), 'r') as f_in:
        for line in f_in:
            tokens = line.strip().split('\t')
            str_to_class[tokens[0]] = tokens[1]
    with open(os.path.join(data_folder, 'str_to_class.pkl'), 'wb') as fp:
        pickle.dump(str_to_class, fp)



    start = time()
    ### Load in training set
    train_image_files = []
    train_label_strs = []
    for root, dirs, files in os.walk(os.path.join(data_folder, 'train')):
        for i, name in enumerate(files):
            if name.endswith('JPEG'):
                train_image_files.append(os.path.join(root, name))
                label = re.match('.*train/(.*)/images$', root).group(1)
                train_label_strs.append(label)

    label_set = list(set(train_label_strs))
    map_str_to_label = {str:i for i, str in enumerate(label_set)}

    with open(os.path.join(data_folder, 'label_set.pkl'), 'wb') as fp:
        pickle.dump(label_set, fp)

    train_images = []
    train_labels = []
    for i in range(len(train_image_files)):
        if i % 10000 == 0:
            print(i, '/', len(train_image_files), time() - start)
        img = imageio.imread(train_image_files[i], as_gray=False, pilmode="RGB")
        img = np.asarray(img)
        assert img.shape == (64, 64, 3), img.shape
        train_images.append(img)
        train_labels.append(map_str_to_label[train_label_strs[i]])
        # print(label_set[map_str_to_label[train_label_strs[i]]], train_image_files[i])


    test_image_files = []
    test_label_strs = []
    # for root, dirs, files in os.walk('imagenet_data/val/images'):
    #     for name in files:
    #         if name.endswith('JPEG'):
    #             test_image_files.append(os.path.join(root, name))


    ### Load in validation set

    with open(os.path.join(data_folder, 'val/val_annotations.txt'), 'r') as f_in:
        for line in f_in:
            tokens = line.strip().split('\t')
            test_image_files.append(os.path.join(data_folder, 'val/images/'+tokens[0]))
            test_label_strs.append(tokens[1])

    test_images = []
    test_labels = []
    for i in range(len(test_image_files)):
        img = imageio.imread(test_image_files[i], as_gray=False, pilmode="RGB")
        img = np.asarray(img)
        assert img.shape == (64, 64, 3), img.shape
        test_images.append(img)
        test_labels.append(map_str_to_label[test_label_strs[i]])
        # print(label_set[map_str_to_label[test_label_strs[i]]], test_image_files[i])


    train_images = np.stack(train_images, axis=0)
    test_images = np.stack(test_images, axis=0)

    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    # plt.imshow(test_images[4])
    # print(test_labels[4])

    np.savez(os.path.join(data_folder, 'train'), images=train_images, labels=train_labels)
    np.savez(os.path.join(data_folder, 'test'), images=test_images, labels=test_labels)


    with open(os.path.join(data_folder, 'label_set.pkl'), "rb") as fp:
        label_set = pickle.load(fp)
    with open(os.path.join(data_folder, 'str_to_class.pkl'), 'rb') as fp:
        str_to_class = pickle.load(fp)
    train = np.load(os.path.join(data_folder, 'train.npz'))
    test = np.load(os.path.join(data_folder, 'test.npz'))

    print(train['images'].shape, train['labels'].shape)
    print(test['images'].shape, test['labels'].shape)
    print(str_to_class[label_set[train['labels'][0]]])
    print(str_to_class[label_set[test['labels'][0]]])

    plt.imshow(train['images'][0])
    plt.show()

    plt.imshow(test['images'][0])
    plt.show()

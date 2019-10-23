'''
Script to save Drebin locally
'''
import numpy as np
import re
import os
import random
from time import time
from scipy.sparse import lil_matrix, csr_matrix, issparse
import multiprocessing as mp


def save_sparse_csr(filename, **kwargs):
    arg_dict = dict()
    for key, value in kwargs.items():
        if issparse(value):
            value = value.tocsr()
            arg_dict[key+'_data'] = value.data
            arg_dict[key+'_indices'] = value.indices
            arg_dict[key+'_indptr'] = value.indptr
            arg_dict[key+'_shape'] = value.shape
        else:
            arg_dict[key] = value

    np.savez(filename, **arg_dict)

def load_sparse_csr(filename):
    loader = np.load(filename)
    new_d = dict()
    finished_sparse_list = []
    sparse_postfix = ['_data', '_indices', '_indptr', '_shape']

    for key, value in loader.items():
        IS_SPARSE = False
        for postfix in sparse_postfix:
            if key.endswith(postfix):
                IS_SPARSE = True
                key_original = re.match('(.*)'+postfix, key).group(1)
                if key_original not in finished_sparse_list:
                    value_original = csr_matrix((loader[key_original+'_data'], loader[key_original+'_indices'], loader[key_original+'_indptr']),
                                      shape=loader[key_original+'_shape'])
                    new_d[key_original] = value_original.tolil()
                    finished_sparse_list.append(key_original)
                break

        if not IS_SPARSE:
            new_d[key] = value

    return new_d

def show_idx(filename):
    map_dict = np.load('feats_map.npz')
    feat_to_idx = map_dict['feat_to_idx'].item()
    idx_list = []

    with open(filename, 'r') as f_in:
        for line in f_in:
            idx_list.append(feat_to_idx[line.strip('\n')])
    return tuple(sorted(idx_list))

def show_feat(idx_list):
    map_dict = np.load('feats_map')
    idx_to_feat = map_dict['idx_to_feat']

    for idx in idx_list:
        print(idx_to_feat[idx])

def get_non_zero_indices(x):
    return tuple(sorted(list(x.nonzero()[0])))


def process_data_sparse_wrapper(q, apps, feats, malwares, ind, count, lock):
    xs, ys = process_data_sparse(apps, feats, malwares, ind)
    q.put((xs, ys, apps))
    with lock:
        count.value += 1
    print('worker', ind, 'put', count.value)

def process_data_multiprocess(apps, feats, malwares, num_of_processes):
    num_of_data = len(apps)
    num_of_features = len(feats)

    x_tot = lil_matrix((num_of_data, num_of_features), dtype=np.int8)
    y_tot = np.zeros(num_of_data)

    frac_num_of_data = np.ceil(num_of_data / num_of_processes).astype(int)

    mp.set_start_method('spawn', force=True)
    count = mp.Value('i', 0)
    lock = mp.Lock()
    q = mp.Queue()
    jobs = []
    buffer = []


    cur_ind = 0
    for i in range(num_of_processes):
        cut_off = np.min([cur_ind+frac_num_of_data, num_of_data])
        frac_apps = apps[cur_ind:cut_off]
        cur_ind = cut_off

        p = mp.Process(target=process_data_sparse_wrapper, args=(q, frac_apps, feats, malwares, i, count, lock))
        jobs.append(p)
        p.start()

    while count.value < num_of_processes:
        continue

    # Start to take from Queue in order to let the threads join. Otherwise, they will block forever.
    while not q.empty():
        buffer.append(q.get())

    for j in jobs:
        j.join()

    cur_ind = 0
    processed_apps_list = []
    for i in range(num_of_processes):
        print('part', i, 'to be merged')

        xs, ys, processed_apps = buffer[i]
        cut_off = cur_ind + xs.shape[0]

        x_tot[cur_ind:cut_off, :] = xs
        y_tot[cur_ind:cut_off] = ys
        processed_apps_list.extend(processed_apps)

        cur_ind = cut_off

    print('finish merging')

    return x_tot, y_tot, processed_apps_list


def process_data_sparse(apps, feats, malwares, ind):
    num_of_data = len(apps)
    num_of_features = len(feats)

    xs = lil_matrix((num_of_data, num_of_features), dtype=np.int8)
    ys = np.zeros(num_of_data)

    start = time()

    for i, app in enumerate(apps):
        if i % 10 == 0:
            print('worker', ind, ', ', i, '/', len(apps), 'time:', time()-start)

        if app in malwares:
            ys[i] = 1  # malware
        else:
            ys[i] = 0  # benign

        with open('../Drebin_data/feature_vectors/' + app, 'r') as f:
            app_feats = [line.strip('\n') for line in f]
            # for feat in app_feats:
            #     if feat in feats:
            #         j = feats.index(feat)
            #         xs[i, j] = 1.
            for j, feat in enumerate(feats):
                if feat in app_feats:
                    xs[i, j] = 1.

    print('worker', ind, 'gets done', num_of_data, 'apps')

    return xs, ys

# def process_data(apps, feats, malwares):
#     xs = []
#     ys = []
#
#     start = time()
#     for i, app in enumerate(apps):
#         if i % 10 == 0:
#             print(i, '/', len(apps), 'time:', time()-start)
#         if app in malwares:
#             ys.append(np.array([1, 0]))  # malware
#         else:
#             ys.append(np.array([0, 1]))  # benign
#         xs.append(preprocess_app(app, feats))
#     xs = np.array(xs)
#     ys = np.array(ys)
#     return xs, ys
#
# def preprocess_app(app, feats):
#     app_vect = np.zeros_like(feats, np.float32)
#     with open('Drebin_data/feature_vectors/' + app, 'r') as f:
#         app_feats = [line.strip('\n') for line in f]
#         for i, feat in enumerate(feats):
#             if feat in app_feats:
#                 app_vect[i] = 1.
#     return app_vect


if __name__ == '__main__':
    if not os.path.isdir('../Drebin_data'):
        os.mkdir('../Drebin_data')

    num_of_processes = 66
    train_test_split_ratio = 0.66

    train_test_apps = None
    malwares = []
    feats = set()

    with open('../Drebin_data/sha256_family.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            malwares.append(line.split(',')[0])
    print('malwares loaded')

    train_test_apps = os.listdir('../Drebin_data/feature_vectors')
    random.seed(0)
    random.shuffle(train_test_apps)
    for filename in train_test_apps:
        with open('../Drebin_data/feature_vectors/' + filename, 'r') as f:
            for line in f:
                feats.add(line.strip('\n'))
    print('features loaded')

    # Save feature mapping to help debugging
    feats = list(feats)
    feat_to_idx = {feat:i for i, feat in enumerate(feats)}
    np.savez('feats_map', idx_to_feat=feats, feat_to_idx=feat_to_idx)
    print('feature mappings saved')


    sensitive_mask = np.zeros(len(feats))
    manifest_features = ['intent', 'permission', 'activity', 'feature', 'provider', 'service_receiver']
    for i, feat in enumerate(feats):
        for manifest_feature in manifest_features:
            fea_len = len(manifest_feature)
            if len(feat) > fea_len and feat[:fea_len] == manifest_feature[:fea_len]:
                # print(feat, manifest_feature)
                sensitive_mask[i] = 1
    print('sensitive mask created')


    train_test_apps = train_test_apps[:int(len(train_test_apps))]

    images, labels, processed_apps_list = process_data_multiprocess(train_test_apps, feats, malwares, num_of_processes)
    print('sensitive_mask sum', np.sum(sensitive_mask))
    print('processed_data shape', images.shape)


    cut_off = int(len(train_test_apps) * train_test_split_ratio)
    train_images, train_labels = images[:cut_off, :], labels[:cut_off]
    test_images, test_labels = images[cut_off:, :], labels[cut_off:]
    save_sparse_csr('../Drebin_data/train', images=train_images, labels=train_labels)
    save_sparse_csr('../Drebin_data/test', images=test_images, labels=test_labels)
    np.save('../Drebin_data/sensitive_mask', sensitive_mask)
    print('data saved')


    with open('../Drebin_data/processed_apps_list.txt', 'w') as f_out:
        for processed_app in processed_apps_list:
            f_out.write(processed_app+'\n')


    print('-'*10, 'TESTING', '-'*10)
    train = load_sparse_csr('../Drebin_data/train.npz')
    test = load_sparse_csr('../Drebin_data/test.npz')

    # np.savez('Drebin_data/train', images=train_images, labels=train_labels)
    # np.savez('Drebin_data/test', images=test_images, labels=test_labels)
    # np.savez('Drebin_data/sensitive_mask', sensitive_mask)
    #
    # train = np.load('Drebin_data/train.npz')


    print('train:', train['images'].shape, train['labels'].shape)
    print('test:', test['images'].shape, test['labels'].shape)
    # print(type(train['images']))
    # print(type(train['images'].item()))
    # print(train['images'].item().shape)

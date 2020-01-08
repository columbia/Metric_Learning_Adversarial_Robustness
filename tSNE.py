import os
import json
import argparse
import copy

from time import time

import tensorflow as tf
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# CPU
from MulticoreTSNE import MulticoreTSNE as TSNE
# from sklearn.manifold import TSNE
# GPU
# from tsnecuda import TSNE

import dataloader.cifar10_input
import dataloader.mnist_input
import dataloader.imagenet_input_new

from learning.model_vanilla import ModelVani
from learning.model_mnist_large import ModelMNIST
# from learning.model_imagenet_wrn import ModelImagenet

from pgd_attack import LinfPGDAttack
from utils import trainable_in, remove_duplicate_node_from_list, l2_norm_reshape

# Parse input parameters
parser = argparse.ArgumentParser(description='Train Triplet')
parser.add_argument('--dataset', dest='dataset', type=str, default='cifar10', help='dataset to use')
args = parser.parse_args()

dataset_type = args.dataset
gpu_options = tf.GPUOptions(allow_growth=True, visible_device_list='0')
precision = tf.float32

config = None
if dataset_type == 'cifar10':
    with open('config_cifar.json') as config_file:
        config = json.load(config_file)
elif dataset_type == 'mnist':
    with open('config_mnist.json') as config_file:
        config = json.load(config_file)
elif dataset_type == 'drebin':
    with open('config_drebin.json') as config_file:
        config = json.load(config_file)
elif dataset_type == 'imagenet':
    with open('config_imagenet.json') as config_file:
        config = json.load(config_file)

strong_attack_config = config['strong_attack']
data_path = config['data_path']
label_smoothing = config['label_smoothing']



def process_data(config, attack_method, num_nat_batches=10, num_adv_batches=10, batch_size=50, model_load_dir='', include_train=False, num_train_nat_batches=10, layer='x4', model_name = 'res50'):
    '''
    Get the x4 layer embeddings of the input data.

    number of natural points = num_nat_batches * batch_size
    number of adversary points = num_adv_batches * batch_size

    raw_dataset and raw_dataset2 provide the same batches of data. The reason they are separated is for the case when we do not want to draw all the adversary points.

    INPUT:
    attack_method: one of ['FGSM', 'PGD20']
    num_nat_batches:
    num_adv_batches:
    batch_size:
    model_load_dir: path to the folder containing the trained model.
    include_train: if also process training samples.
    num_train_nat_batches:
    OUTPUT:
    natural_embed: 2d numpy array with size [number of natural points, embedding size]
    natural_ypred: 2d numpy array with size [number of natural points,]
    natural_y: 2d numpy array with size [number of natural points,]
    adversary_embed: 2d numpy array with size [number of adversary points, embedding size]
    adversary_ypred: 2d numpy array with size [number of adversary points,]
    adversary_y: 2d numpy array with size [number of adversary points,]
    train_natural_embed:
    train_natural_ypred:
    train_natural_y:
    layer:
    '''

    input_shape, raw_dataset, raw_dataset2, model = None, None, None, None
    eps = config["epsilon"]

    if dataset_type == 'cifar10':
        input_shape = [None, 32, 32, 3]
        raw_dataset = dataloader.cifar10_input.CIFAR10Data(data_path)
        model = ModelVani(precision=precision)
    elif dataset_type == 'mnist':
        input_shape = [None, 28, 28]
        raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)
        model = ModelMNIST(precision=precision)
    elif dataset_type == 'imagenet':
        input_shape = [None, 64, 64, 3]
        raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)

        eps = config["epsilon"]

        if model_name.startswith('res101'):
            from learning.model_imagenet_res101 import ModelImagenet
            model = ModelImagenet(0)
        elif model_name.startswith('res50'):
            from learning.model_imagenet_res50 import ModelImagenet
            model = ModelImagenet(0)
        elif model_name.startswith('res59'):
            from learning.model_imagenet_res59 import ModelImagenet
            model = ModelImagenet(0)



    raw_dataset2 = copy.deepcopy(raw_dataset)

    with tf.variable_scope('input'):
        x_Anat = tf.placeholder(precision, shape=input_shape)
        x_Aadv = tf.placeholder(precision, shape=input_shape)
        y_Ainput = tf.placeholder(tf.int64, shape=None)
        is_training = tf.placeholder(tf.bool, shape=None)


    # A
    layer_values_A, _, _, _, _, n_Aaccuracy, n_Apredict, _ = model._encoder(x_Anat, y_Ainput, is_training)
    # A'
    layer_values_Ap, a_Axent, _, _, _, a_Aaccuracy, a_Apredict, _ = model._encoder(x_Aadv, y_Ainput, is_training, mask_effective_attack=config['mask_effective_attack'])


    assert layer in ['x1', 'x2', 'x3', 'x4']

    f_x4_nat = l2_norm_reshape(layer_values_A[layer])
    f_x4_adv = l2_norm_reshape(layer_values_Ap[layer])


    model_var_attack = x_Aadv, a_Axent, y_Ainput, is_training, a_Aaccuracy

    PGD20 = LinfPGDAttack(model_var_attack,
                          eps,
                          strong_attack_config[0],
                          strong_attack_config[1],
                          config['random_start'],
                          'xent',
                          dataset_type
                          )

    FGSM = LinfPGDAttack(model_var_attack,
                         config['epsilon'],
                         1,
                         config['epsilon'],
                         config['random_start'],
                         'xent',
                         dataset_type
                         )

    attack_model = None
    if attack_method == 'PGD20':
        attack_model = PGD20
    elif attack_method == 'FGSM':
        attack_model = FGSM
    else:
        raise

    natural_embed = []
    natural_ypred = []
    natural_y = []
    adversary_embed = []
    adversary_ypred = []
    adversary_y = []
    train_natural_embed = []
    train_natural_ypred = []
    train_natural_y = []


    var_main_encoder = trainable_in('main_encoder')
    var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
    restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
    saver_restore = tf.train.Saver(restore_var_list)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        sess.run(tf.global_variables_initializer())

        print('dir to load :', model_load_dir)
        model_dir_load = tf.train.latest_checkpoint(model_load_dir)
        saver_restore.restore(sess, model_dir_load)



        if include_train:
            for ii in range(num_train_nat_batches):
                x_batch, y_batch = raw_dataset.train_data.get_next_batch(batch_size)

                if dataset_type == 'imagenet':
                    x_batch = x_batch.astype(np.float32)

                A_dict = {x_Anat: x_batch.astype(np.float32), y_Ainput: y_batch, is_training: False}
                f_x4_nat_eval, nat_acc, nat_ypred = sess.run([f_x4_nat, n_Aaccuracy, n_Apredict], feed_dict=A_dict)

                train_natural_embed.append(f_x4_nat_eval)
                train_natural_ypred.append(nat_ypred)
                train_natural_y.append(y_batch)

                print('Train Natural', ii, '/', num_train_nat_batches, nat_acc)


        for ii in range(num_nat_batches):
            x_batch, y_batch = raw_dataset.eval_data.get_next_batch(batch_size)

            if dataset_type == 'imagenet':
                x_batch = x_batch.astype(np.float32)

            A_dict = {x_Anat: x_batch.astype(np.float32), y_Ainput: y_batch, is_training: False}


            f_x4_nat_eval, nat_acc, nat_ypred = sess.run([f_x4_nat, n_Aaccuracy, n_Apredict], feed_dict=A_dict)

            natural_embed.append(f_x4_nat_eval)
            natural_ypred.append(nat_ypred)
            natural_y.append(y_batch)

            print('Test Natural', ii, '/', num_nat_batches, nat_acc)
            # print(nat_ypred, y_batch)

        for ii in range(num_adv_batches):
            x_batch, y_batch = raw_dataset2.eval_data.get_next_batch(batch_size)

            if dataset_type == 'imagenet':
                x_batch = x_batch.astype(np.float32)

            x_batch_adv = attack_model.perturb(x_batch, y_batch, False, sess)

            Aattack_dict = {x_Aadv: x_batch_adv.astype(np.float32), y_Ainput: y_batch, is_training: False}
            f_x4_adv_eval, adv_acc, adv_ypred = sess.run([f_x4_adv, a_Aaccuracy, a_Apredict], feed_dict=Aattack_dict)

            # from utils import visualize_imgs
            # print(x_batch)
            # print(x_batch_adv)
            # visualize_imgs('../mnist_imgs/', [x_batch, x_batch_adv, x_batch_adv-x_batch], img_ind=ii)

            adversary_embed.append(f_x4_adv_eval)
            adversary_ypred.append(adv_ypred)
            adversary_y.append(y_batch)

            print('Test Adversary', ii, '/', num_adv_batches, adv_acc)
            # print(adv_ypred, y_batch)





        print('Begin Concatenation')
        if include_train:
            train_natural_embed = np.concatenate(train_natural_embed, axis=0)
            train_natural_ypred = np.concatenate(train_natural_ypred, axis=0)
            train_natural_y = np.concatenate(train_natural_y, axis=0)

        natural_embed = np.concatenate(natural_embed, axis=0)
        natural_ypred = np.concatenate(natural_ypred, axis=0)
        natural_y = np.concatenate(natural_y, axis=0)
        adversary_embed = np.concatenate(adversary_embed, axis=0)
        adversary_ypred = np.concatenate(adversary_ypred, axis=0)
        adversary_y = np.concatenate(adversary_y, axis=0)
        print('Finish Concatenation')

        print('avg test nat acc', np.mean(natural_ypred==natural_y))
        print('avg test adv acc', np.mean(adversary_ypred==adversary_y))
        with open('tmp.txt', 'a') as f:
            f.write(model_load_dir+'\n')
            f.write('avg test nat acc :'+str(np.mean(natural_ypred==natural_y))+'\n')
            f.write('avg test adv acc :'+str(np.mean(adversary_ypred==adversary_y))+'\n')

        return natural_embed, natural_ypred, natural_y, adversary_embed, adversary_ypred, adversary_y, train_natural_embed, train_natural_ypred, train_natural_y



def run_tSNE(natural_embed, adversary_embed, n_jobs, perplexity):
    '''
    Need CUDA 9.0 and install the tsnecuda package by running
    conda install tsnecuda -c cannylab

    Apply t-SNE to the input data
    INPUT:
        natural_embed: 2d numpy array with size [number of points, embedding length]
        adversary_embed: 2d numpy array with size [number of points, embedding length]
        n_jobs:
        perplexity:
    OUTPUT:
        natural_2d: 2d numpy array with size [number of points, 2]
        adversary_2d: 2d numpy array with size [number of points, 2]
    '''

    cutoff = natural_embed.shape[0]
    X = np.concatenate([natural_embed, adversary_embed], axis=0)

    # CPU Sklearn
    # tsne = TSNE(perplexity=perplexity, n_iter=5000, n_iter_without_progress=800, learning_rate=20, metric='cosine')
    # X_embedded = tsne.fit_transform(X)

    # CPU
    tsne = TSNE(n_jobs=n_jobs, perplexity=perplexity, n_iter=5000, n_iter_without_progress=800, learning_rate=20, metric='cosine')
    X_embedded = tsne.fit_transform(X)

    # GPU
    # X_embedded = TSNE(n_components=2, perplexity=30, learning_rate=10).fit_transform(X)

    return X_embedded[:cutoff], X_embedded[cutoff:]


def visualize_tSNE(with_special_legend, ax, dataset_type, use_single_cluster, single_cluster_ind, use_double_clusters, double_cluster_inds, image_name, natural_2d, adversary_2d, natural_ypred, adversary_ypred, natural_y, adversary_y):
    '''
    INPUT:
        with_special_legend:
        ax: ax object for plotting
        dataset_type: dataset to use
        use_single_cluster: for reproducing Figure 1
        single_cluster_ind: the index of the class to visualize
        use_double_clusters: for reproducing Figure 2
        double_cluster_inds: the indices of the two classes to visualize
        image_name: name of the generated image
        natural_2d: 2d numpy array with size [number of points, 2]
        adversary_2d: 2d numpy array with size [number of points, 2]
        natural_ypred: 2d numpy array with size [number of points,]
        adversary_ypred: 2d numpy array with size [number of points,]
        natural_y: 2d numpy array with size [number of points,]
        adversary_y: 2d numpy array with size [number of points,]
    '''


    image_name = str(image_name)
    cm = plt.get_cmap('Paired')
    num_colors = 10
    if dataset_type == 'cifar10':
        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif dataset_type == 'mnist':
        label_names = [str(i) for i in range(10)]
    single_ind_label_names = None
    double_ind_label_names = None
    if use_single_cluster:
        single_ind_label_names = label_names[single_cluster_ind]
    elif use_double_clusters:
        double_ind_label_names = [label_names[i] for i in double_cluster_inds]
    colors = [cm(1.*i/num_colors) for i in range(num_colors)]
    colors_single = cm(1.*single_cluster_ind/num_colors)
    single_cluster_color = 'gainsboro'


    correct_nat = natural_2d[natural_y == natural_ypred]

    print('Correct Nat:', len(correct_nat), '/', len(natural_y))


    if use_double_clusters:
        # print(natural_y[:100])
        # print(colors_double[0], colors_double[1])
        natural_colors = ['green', 'blue']
        inds_0 = natural_y==double_cluster_inds[0]
        ax.scatter(natural_2d[inds_0][:, 0], natural_2d[inds_0][:, 1], c=natural_colors[0], s=40)

        inds_1 = natural_y==double_cluster_inds[1]
        ax.scatter(natural_2d[inds_1][:, 0], natural_2d[inds_1][:, 1], c=natural_colors[1], s=40)

        h_natural = [plt.plot([],[], color=natural_colors[0], marker="o", ls="", markersize=30)[0], plt.plot([],[], color=natural_colors[1], marker="o", ls="", markersize=30)[0]]

        legend_natural = ax.legend(handles=h_natural, labels=double_ind_label_names, loc=1, framealpha=0.3, prop={'size': 30})
        legend_natural.set_title("natural", prop = {'size':30})
        ax.add_artist(legend_natural)
    elif use_single_cluster:
        ax.scatter(natural_2d[:, 0], natural_2d[:, 1], c=single_cluster_color, s=40)

        if with_special_legend:
            wrong_inds = list(range(len(colors)))
            wrong_inds.remove(single_cluster_ind)

            del label_names[single_cluster_ind]

            h_natural = [plt.plot([],[], color=single_cluster_color, marker="o", ls="", markersize=15)[0]]

            correct_h_single_ind = [plt.plot([],[], color=colors_single, marker=">", ls="", markersize=15)[0]]

            wrong_h_single_ind = [plt.plot([],[], color=colors[i], marker="<", ls="", markersize=15)[0] for i in wrong_inds]

            h_single_ind = h_natural + correct_h_single_ind + wrong_h_single_ind

            legend_all = ax.legend(handles=h_single_ind, labels=[single_ind_label_names+'(clean)']+[single_ind_label_names+'(adv)']+label_names, loc=2, framealpha=0.3, prop={'size': 15})

            legend_all.set_title("Prediction", prop = {'size':17})


            h_natural = [plt.plot([],[], color=single_cluster_color, marker="o", ls="", markersize=10)[0]]
            legend_natural = ax.legend(handles=h_natural, labels=[single_ind_label_names], loc=1, framealpha=0.3, prop={'size': 13})
            legend_natural.set_title("natural", prop = {'size':13})
            ax.add_artist(legend_natural)

    if use_double_clusters:
        correct_adv_color = 'orange'
        wrong_adv_color = 'red'

        correct_adv = adversary_2d[adversary_ypred == double_cluster_inds[0]]
        correct_adv_labels = adversary_y[adversary_ypred == double_cluster_inds[0]]
        ax.scatter(correct_adv[:, 0], correct_adv[:, 1], c=correct_adv_color, marker='<', s=40)

        wrong_adv = adversary_2d[adversary_ypred == double_cluster_inds[1]]
        wrong_adv_labels = adversary_y[adversary_ypred == double_cluster_inds[1]]
        ax.scatter(wrong_adv[:, 0], wrong_adv[:, 1], c=wrong_adv_color, marker='<', s=200)

        h_wrong_adv = [plt.plot([],[], color=wrong_adv_color, marker="<", ls="", markersize=30)[0]]

        legend_wrong_adv = ax.legend(handles=h_wrong_adv, labels=[double_ind_label_names[0]], fontsize = 'large', loc=2, framealpha=0.3, prop={'size': 30})
        legend_wrong_adv.set_title("misclassified adv", prop = {'size':30})
        ax.add_artist(legend_wrong_adv)
    elif use_single_cluster:

            correct_h_single_ind = [plt.plot([],[], color=colors_single, marker=">", ls="", markersize=10)[0]]
            correct_legend_adv = ax.legend(handles=correct_h_single_ind, labels=[single_ind_label_names], loc=2, framealpha=0.3, prop={'size': 13})
            correct_legend_adv.set_title("correct adversary", prop = {'size': 13})
            ax.add_artist(correct_legend_adv)

            wrong_h_single_ind = [plt.plot([],[], color=colors[i], marker="<", ls="", markersize=10)[0] for i in range(len(colors))]
            wrong_legend_adv = ax.legend(handles=wrong_h_single_ind, labels=label_names, loc=3, framealpha=0.3, prop={'size': 13})
            wrong_legend_adv.set_title("wrong adversary", prop = {'size':13})
            ax.add_artist(wrong_legend_adv)

            correct_adv = adversary_2d[adversary_y == adversary_ypred]
            correct_adv_labels = adversary_y[adversary_y == adversary_ypred]
            ax.scatter(correct_adv[:, 0], correct_adv[:, 1], c=correct_adv_labels, cmap=matplotlib.colors.ListedColormap(colors_single), marker='>', s=40)

            wrong_adv = adversary_2d[adversary_y != adversary_ypred]
            wrong_adv_labels = adversary_ypred[adversary_y != adversary_ypred]
            ax.scatter(wrong_adv[:, 0], wrong_adv[:, 1], c=wrong_adv_labels, cmap=matplotlib.colors.ListedColormap(colors), marker='<', s=40)










if __name__ == '__main__':
    # ['full', 'after preprocess', 'after tsne']
    # full is by default. It creates embeddings, run t-SNE and generate plots
    # after preprocess only run t-SNE and generate plots
    # after tsne only generate plots
    running_mode = 'full'
    # ['FGSM', 'PGD20']
    attack_method = 'PGD20'
    # which models to draw: ['natural', 'madry', 'alp', 'triplet']
    models = ['natural', 'madry', 'alp', 'triplet']
    # which layer to use: ['x1', 'x2', 'x3', 'x4']
    layer = 'x4'

    # reproduce Figure 1 and Figure 2, respectively
    # 'single_cluser', 'double_cluster'
    mode = 'single_cluster'
    # This needs to be changed to the folder contains the model's folder
    # and the model's folder needs to have the names 'natural', 'madry', 'alp',
    # 'triplet', respectively
    model_folder_dir = '../../19Fall/load/cifar10/'

    # use fractional of testing natural examples for better visualization
    use_fractional = True
    # the number of testing natural examples used
    sample_num_for_single = 500
    sample_num_for_double = 200
    # class index/indices
    single_cluster_ind = 4
    double_cluster_inds = [9, 2]
    use_single_cluster = False
    use_double_clusters = False
    if mode == 'single_cluster':
        use_single_cluster = True
        use_fractional = True
    elif mode == 'double_cluster':
        use_double_clusters = True


    # what kinds of adversary points are included
    only_correct_adv = False
    only_wrong_adv = True
    # if drawing separate plots for each model
    draw_separate = False

    # number of threads used
    n_jobs = 16
    # parameter for t-SNE
    perplexity = 15


    # number of bataches in the testing set to consider
    num_nat_batches = 200
    num_adv_batches = 200
    batch_size = 50





    root_folder_name = 'tsne'
    folder_name = root_folder_name+'/'+dataset_type+'/'+layer
    tsne_folder_name = folder_name
    if not os.path.exists(root_folder_name):
        os.mkdir(root_folder_name)
    if not os.path.exists(root_folder_name+'/'+dataset_type):
        os.mkdir(root_folder_name+'/'+dataset_type)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    preprocess_data_prefix = dataset_type + '_data_preprocess_'
    tsne_data_prefix = dataset_type + '_data_'
    original_tsne_folder_name = tsne_folder_name

    start = time()

    # Temporarily change for double inds
    if use_single_cluster:
        tsne_folder_name = original_tsne_folder_name + '/' + str(single_cluster_ind)
    elif use_double_clusters:
        tsne_folder_name = original_tsne_folder_name + '/'+str(double_cluster_inds[0])+'_'+str(double_cluster_inds[1])
    if not os.path.exists(tsne_folder_name):
        os.mkdir(tsne_folder_name)


    if not draw_separate:
        fig, axs = plt.subplots(1, len(models), figsize=(len(models)*10,10), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    for ind, model in enumerate(models):
        if draw_separate:
            fig, axs = plt.subplots(1, 1, figsize=(10,10), sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
            ax = axs
        if not draw_separate:
            if len(models) > 1:
                ax = axs[ind]
            else:
                ax = axs

        print('-'*10, model, '-'*10)
        # Setup
        outfile = tsne_data_prefix + model
        path_tsne = tsne_folder_name +'/'+outfile

        outfile_preprocess = preprocess_data_prefix + model
        path_preprocess = folder_name+'/'+outfile_preprocess

        model_load_dir = os.path.join(model_folder_dir, model)


        # Preprocess
        if running_mode == 'full':
            natural_embed, natural_ypred, natural_y, adversary_embed, adversary_ypred, adversary_y, _, _, _ = process_data(config, attack_method=attack_method, num_nat_batches=num_nat_batches, num_adv_batches=num_adv_batches, batch_size=batch_size, model_load_dir=model_load_dir, layer=layer)

            saved_data_preprocess = {'natural_embed': natural_embed, 'adversary_embed': adversary_embed, 'natural_ypred': natural_ypred, 'adversary_ypred': adversary_ypred, 'natural_y': natural_y, 'adversary_y': adversary_y}
            np.savez(path_preprocess, **saved_data_preprocess)
            print(time()-start, 'Finish Processing', natural_embed.shape, adversary_embed.shape)



        # t-SNE
        if running_mode in ['full', 'after preprocess']:
            loaded_data_preprocess = np.load(path_preprocess+'.npz')
            natural_embed = loaded_data_preprocess['natural_embed']
            adversary_embed = loaded_data_preprocess['adversary_embed']
            natural_ypred = loaded_data_preprocess['natural_ypred']
            adversary_ypred = loaded_data_preprocess['adversary_ypred']
            natural_y = loaded_data_preprocess['natural_y']
            adversary_y = loaded_data_preprocess['adversary_y']


            # Select two clusters with adversary of the first cluster.
            if use_double_clusters:
                natural_selected_ind = natural_y == double_cluster_inds[0]
                natural_selected_ind |= natural_y == double_cluster_inds[1]
                natural_embed = natural_embed[natural_selected_ind]
                natural_ypred = natural_ypred[natural_selected_ind]
                natural_y = natural_y[natural_selected_ind]

                if use_fractional:
                    natural_embed = natural_embed[:sample_num_for_double]
                    natural_ypred = natural_ypred[:sample_num_for_double]
                    natural_y = natural_y[:sample_num_for_double]

                adversary_selected_ind = adversary_y == double_cluster_inds[0]
                if only_wrong_adv:
                    adversary_selected_ind &= adversary_ypred == double_cluster_inds[1]
                elif only_correct_adv:
                    adversary_selected_ind &= adversary_ypred == double_cluster_inds[0]
                N = np.sum(adversary_selected_ind)
                adversary_embed = adversary_embed[adversary_selected_ind]
                adversary_ypred = adversary_ypred[adversary_selected_ind]
                adversary_y = adversary_y[adversary_selected_ind]

            elif use_single_cluster:

                natural_selected_ind = natural_y == single_cluster_ind

                natural_embed = natural_embed[natural_selected_ind]
                natural_ypred = natural_ypred[natural_selected_ind]
                natural_y = natural_y[natural_selected_ind]

                if use_fractional:
                    natural_embed = natural_embed[:sample_num_for_single]
                    natural_ypred = natural_ypred[:sample_num_for_single]
                    natural_y = natural_y[:sample_num_for_single]


            selected_cluster_ind = single_cluster_ind
            if use_double_clusters:
                selected_cluster_ind = double_cluster_inds[0]
            adversary_selected_ind = adversary_y == selected_cluster_ind

            adversary_embed = adversary_embed[adversary_selected_ind]
            adversary_ypred = adversary_ypred[adversary_selected_ind]
            adversary_y = adversary_y[adversary_selected_ind]


            natural_2d, adversary_2d = run_tSNE(natural_embed, adversary_embed, n_jobs, perplexity)
            print(time()-start, 'Finish tSNE')

            saved_data_tsne = {'image_name': path_tsne, 'natural_2d': natural_2d, 'adversary_2d': adversary_2d, 'natural_ypred': natural_ypred, 'adversary_ypred': adversary_ypred, 'natural_y': natural_y, 'adversary_y': adversary_y}
            np.savez(path_tsne, **saved_data_tsne)
            print('Finish saving')

        if running_mode in ['full', 'after preprocess', 'after tsne']:
            # Visualize

            with_special_legend = True
            if model == 'natural':

                with_special_legend = False
            loaded_data_tsne = np.load(path_tsne+'.npz')
            visualize_tSNE(with_special_legend, ax, dataset_type, use_single_cluster, single_cluster_ind, use_double_clusters, double_cluster_inds, **loaded_data_tsne)

            if draw_separate:
                adv_mode = ''
                if only_correct_adv:
                    adv_mode = 'correct'
                elif only_wrong_adv:
                    adv_mode = 'wrong'

                inds_str = ''
                if use_single_cluster:
                    inds_str = '_'+str(single_cluster_ind)+'_'
                elif use_double_clusters:
                    inds_str = '_'+str(double_cluster_inds[0])+'_'+str(double_cluster_inds[1])+'_'

                fig.savefig('tsne/'+adv_mode+inds_str +model+'.pdf', dpi=1000, bbox_inches = 'tight')
                plt.close(fig)

    if running_mode in ['full', 'after preprocess', 'after tsne']:
        if not draw_separate:
            if len(models) > 1:
                for ax in axs.flat:
                    ax.label_outer()
            adv_mode = ''

            if only_correct_adv:
                adv_mode = 'correct'
            elif only_wrong_adv:
                adv_mode = 'wrong'

            inds_str = ''
            if use_single_cluster:
                inds_str = '_'+str(single_cluster_ind)+'_'
            elif use_double_clusters:
                inds_str = '_'+str(double_cluster_inds[0])+'_'+str(double_cluster_inds[1])+'_'

            fig.savefig('tsne/'+adv_mode+inds_str+'.pdf', dpi=1000, bbox_inches = 'tight')
            plt.close(fig)

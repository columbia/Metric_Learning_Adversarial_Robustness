'''This function can conduct diffeerent Attacks and plot the loss landscape of a image,
 but I think loss landscape should be focused on TinyImagenet'''
import json
import os
import tensorflow as tf
import numpy as np

from learning.model_vanilla import ModelVani
from learning.model_mnist_large import ModelMNIST
import dataloader.cifar10_input
import dataloader.mnist_input

from pgd_attack import LinfPGDAttack

from utils import remove_duplicate_node_from_list, trainable_in
import math

gpu_options = tf.GPUOptions(allow_growth=True)


def get_latest_checkpoint(train_dir):
  ckpt_files = [train_dir + '/' + fname for fname in os.listdir(train_dir) if 'ckpt' in fname]
  last_mtime, last_fname = 0.0, ''
  for cur_fname in ckpt_files:
    cur_mtime = os.stat(cur_fname).st_mtime  # modification time
    if cur_mtime > last_mtime:
      last_mtime = cur_mtime
      last_fname = cur_fname
  last_fname = last_fname.replace('.meta', '').replace('.index', '').replace('.data-00000-of-00001', '')
  return last_fname



def test_black_attack(model_dir, model_var_attack, n_Anum_correct, var_main_encoder, attack_steps, attack_step_size, config,
               dataset_type, dataset, n_mask, x4, pre_softmax, loss_func='xent', rand_start=1, use_rand=True, momentum=0.0, load_filename=None):
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_Anat, n_Axent, y_Ainput, is_training, n_Aaccuracy = model_var_attack

    num_of_classes = 10
    print('dataset type', dataset_type)
    if dataset_type == 'imagenet':
        saver = tf.train.Saver()
        num_of_classes = 200
        var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
        restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
        saver_restore = tf.train.Saver(restore_var_list)
    elif dataset_type == 'imagenet_01':
        saver = tf.train.Saver()
        num_of_classes = 200
    else:
        var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
        restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
        saver_restore = tf.train.Saver(restore_var_list)

    total_corr_adv = 0.
    total_xent_adv = 0.

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        if dataset_type == 'imagenet_01':
            saver.restore(sess, get_latest_checkpoint(model_dir))
        else:
            model_dir_load = tf.train.latest_checkpoint(model_dir)
            saver_restore.restore(sess, model_dir_load)

        black_attack = np.load(load_filename)[()]
        print("****\nLoading transfer attack\n*******")
        image_per = black_attack['img']
        label = black_attack['label']

        for ibatch in range(num_batches):
            bstart = ibatch * eval_batch_size
            bend = min(bstart + eval_batch_size, num_eval_examples)

            x_batch = image_per[bstart:bend, :]
            y_batch = label[bstart:bend]

            test_dict = {x_Anat: x_batch.astype(np.float32),
                         y_Ainput: y_batch,
                         is_training: False}
            cur_corr_adv, cur_xent_adv = sess.run([n_Anum_correct, n_Axent], feed_dict=test_dict)

            total_xent_adv += cur_xent_adv
            total_corr_adv += cur_corr_adv

    num_batches = (ibatch + 1)
    avg_xent_adv = total_xent_adv / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    print('***TEST**step={}  step_size={}  **'.format(attack_steps, attack_step_size))
    print('Accuracy: {:.2f}%'.format(100 * acc_adv))
    print('loss: {:.4f}'.format(avg_xent_adv))
    print("*****")




def test_model(model_dir, model_var_attack, n_Anum_correct, var_main_encoder, attack_steps, attack_step_size, config,
               dataset_type, dataset, n_mask, x4, pre_softmax, loss_func='xent', rand_start=1, use_rand=True,
               momentum=0.0, save_filename=None):

    num_eval_examples = config['num_eval_examples']  #TODO: for cifar!
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_Anat, n_Axent, y_Ainput, is_training, n_Aaccuracy = model_var_attack

    num_of_classes = 10
    if dataset_type == 'imagenet_01':
        saver = tf.train.Saver()
        num_of_classes = 200
    else:
        var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
        restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
        saver_restore = tf.train.Saver(restore_var_list)
        if dataset_type == 'imagenet':
            num_of_classes = 200

    print('epsilon', config['epsilon'], 'step size', attack_step_size)
    attack = LinfPGDAttack(model_var_attack,
                                  config['epsilon'],
                                  attack_steps,
                                  attack_step_size,
                                  use_rand,
                                  loss_func,
                                  dataset_type,
                                  pre_softmax= pre_softmax,
                                  num_of_classes=num_of_classes,
                           momentum=momentum
                                  )  # TODO: without momentum



    def compute_pred_rep(dataset_part, save_emb, num_batches):

        total_corr_adv = 0
        total_xent_adv = 0.

        ibatch=0

        save_flag = False
        if save_emb != None:
            perturbed_img = []
            gt_label = []
            save_flag=True

        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            if dataset_type == 'imagenet_01':
                saver.restore(sess, get_latest_checkpoint(model_dir))
            else:
                model_dir_load = tf.train.latest_checkpoint(model_dir)
                print('model_dir_load', model_dir_load)
                saver_restore.restore(sess, model_dir_load)

            for ibatch in range(num_batches):
                bstart = ibatch * eval_batch_size
                bend = min(bstart + eval_batch_size, num_eval_examples)

                x_batch = dataset_part.xs[bstart:bend, :]
                if dataset_type == 'imagenet_01':
                    x_batch = x_batch.astype(np.float32) / 255.0
                y_batch = dataset_part.ys[bstart:bend]

                if attack_steps >0:
                    if rand_start > 1:
                        for ii in range(rand_start):
                            x_batch_adv = attack.perturb(x_batch, y_batch, False, sess)
                            if ii == 0:
                                x_adv_final = np.copy(x_batch_adv)
                            test_dict_temp = {x_Anat: x_batch_adv.astype(np.float32),
                                         y_Ainput: y_batch,
                                         is_training: False}
                            n_mask_val = sess.run(n_mask, feed_dict = test_dict_temp)
                            for ind in range(n_mask_val.shape[0]):
                                if n_mask_val[ind]<1e-4:
                                    x_adv_final[ind] = x_batch_adv[ind]
                        x_batch_adv = x_adv_final
                    else:
                        x_batch_adv = attack.perturb(x_batch, y_batch, False, sess)


                else:
                    x_batch_adv = x_batch

                if save_flag:
                    perturbed_img.append(x_batch_adv)
                    gt_label.append(y_batch)

                # print('range', np.max(x_batch_adv), np.min(x_batch_adv))

                test_dict = {x_Anat: x_batch_adv.astype(np.float32),
                             y_Ainput: y_batch,
                             is_training: False}
                cur_corr_adv, cur_xent_adv = sess.run([n_Anum_correct, n_Axent], feed_dict=test_dict)

                # from utils import visualize_imgs
                # # print(x_batch)
                # # print(x_batch_adv)
                # print(np.max(x_batch_adv), np.min(x_batch_adv), np.sum(np.abs(x_batch_adv-x_batch)))
                # visualize_imgs('/home/mcz/AdvPlot/', [x_batch, x_batch_adv, x_batch_adv-x_batch + 0.5], img_ind=ibatch)

                total_xent_adv += cur_xent_adv
                total_corr_adv += cur_corr_adv

                if ibatch %(num_batches//10) == 0:
                    print(ibatch, 'finished')

        if save_flag:
            perturbed_img_all = np.concatenate(perturbed_img, axis=0)
            gt_label_all = np.concatenate(gt_label, axis=0)
            save_dict=dict()
            save_dict['img'] = perturbed_img_all
            save_dict['label'] = gt_label_all
            np.save(save_emb, save_dict)
            print('save successfully')

        num_batches = (ibatch + 1)
        avg_xent_adv = total_xent_adv / num_eval_examples
        acc_adv = total_corr_adv / num_eval_examples

        print('***TEST**step={}  step_size={}  **'.format(attack_steps, attack_step_size))
        print('Accuracy: {:.2f}%'.format(100 * acc_adv))
        print('loss: {:.4f}'.format(avg_xent_adv))
        print("*****")


    compute_pred_rep(dataset.eval_data, save_filename, num_batches)





def one_test(dataset_type, model_load_direction, attack_steps, attack_step_size, loss_func, rand_start=1,
             use_rand=True, model_name=None, momentum=0.0,save_filename=None, black_attack=False,
             vis_lossland_scape=False, model_type='ConvNet'):

    # dataset_type = 'cifar10'
    # model_load_direction = 'models/model_0_renamed'
    # model_load_direction = '/mnt/md0/FSRobust/cifar_models/triplet/switch_adv_only/cifar10,A_Ap_B,A1_Ap_B_1,'

    # model_load_direction ='/mnt/md0/FSRobust/cifar_models/triplet/April15/switch_adv_only_hardneg_mar0.03_lam10/cifar10,A_Ap_B,A1_Ap_B_1,'
    # model_load_direction = '/mnt/md0/FSRobust/cifar_models/triplet/backup/ml2_only/cifar10,A_Ap_B,A1_Ap_B_1,_0'

    # dataset_type = 'mnist'
    # model_load_direction = 'mnist_models/reproduce-secret'
    # model_load_direction = '/mnt/md0/FSRobust/mnist_models/April2/new_schedule_multilayer/mnist,A_Ap_B,A1_Ap_B_1,' #93.21%

    # model_load_direction = '/mnt/md0/FSRobust/mnist_models/April2/ml2_only_train_both/mnist,A_Ap_B,A1_Ap_B_1,'  #ALP l2


    precision = tf.float32

    model = None
    input_shape = None


    if dataset_type == 'cifar10':
        input_shape = [None, 32, 32, 3]
        with open('config_cifar.json') as config_file:
            config = json.load(config_file)
        data_path = config['data_path']

        from learning.model_vanilla import ModelVani
        if model_type == 'Res20':
            from learning.model_cifar10_resnet import CifarResNet
            model = CifarResNet(precision=precision, ratio=config['mask_ratio'])
        elif model_type == 'ConvNet':
            from learning.convnet_cifar import CifarConvNet
            model = CifarConvNet(precision=precision, ratio=config['mask_ratio'])
        elif model_type == 'Res50':
            from learning.model_cifar10_resnet import CifarResNet
            model = CifarResNet(precision=precision, ratio=config['mask_ratio'], mode='50')
        elif model_type == 'Res101':
            from learning.model_cifar10_res101 import CifarResNetUpdate
            model = CifarResNetUpdate(precision=precision, ratio=config['mask_ratio'], mode='101')
        else:
            model = ModelVani(precision=precision)

        raw_dataset = dataloader.cifar10_input.CIFAR10Data(data_path)

    elif dataset_type == 'mnist':
        with open('config_mnist.json') as config_file:
            config = json.load(config_file)
        input_shape = [None, 28, 28]
        if config['model_type'] == 'MLP':
            from learning.model_mnist_mlp import ModelMNISTMLP
            model = ModelMNISTMLP(precision=precision, ratio=config['mask_ratio'])
        else:
            model = ModelMNIST(precision=precision)

        data_path = config['data_path']
        raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)

    elif dataset_type == 'imagenet':
        with open('config_imagenet.json') as config_file:
            config = json.load(config_file)

        # config["epsilon"] = config["epsilon"] / 255.0
        # config["step_size"] = config["step_size"] / 255.0

        input_shape = [None, 64, 64, 3]
        raw_dataset = dataloader.mnist_input.MNISTData(config['tiny_imagenet_data_dir_np'], dataset="imagenet")
        if black_attack:
            model_type = model_type
        else:
            model_type = config['model_type']

        if model_type == 'Res20':
            from learning.model_imagenet_res20 import ModelImagenet
            model = ModelImagenet(batch_size=None, precision=precision, label_smoothing=0.1)
        elif model_type == 'Res50':
            from learning.model_imagenet_res50 import ModelImagenet
            model = ModelImagenet(batch_size=None, precision=precision, label_smoothing=0.1)

    elif dataset_type == 'imagenet_01':
        with open('config_imagenet.json') as config_file:
            config = json.load(config_file)
        input_shape = [None, 64, 64, 3]
        raw_dataset = dataloader.mnist_input.MNISTData(config['tiny_imagenet_data_dir_np'], dataset="imagenet")

        if model_name.startswith('res101'):
            from learning.model_imagenet_res101 import ModelImagenet
            model = ModelImagenet(0)
            config["epsilon"] = config["epsilon"] / 255.0
        elif model_name.startswith('res50'):
            from learning.model_imagenet_res50 import ModelImagenet
            model = ModelImagenet(0)
            config["epsilon"] = config["epsilon"] / 255.0


    x_Anat = tf.placeholder(precision, shape=input_shape)
    y_Ainput = tf.placeholder(tf.int64, shape=None)
    is_training = tf.placeholder(tf.bool, shape=None)

    layer_values_A, n_Axent, n_Amean_xent, _, n_Anum_correct, n_Aaccuracy, _, n_mask = model._encoder(x_Anat, y_Ainput,
                                                                                                 is_training)
    xent_loss = model.y_xent

    model_var_attack = x_Anat, n_Axent, y_Ainput, is_training, n_Aaccuracy
    var_main_encoder = trainable_in('main_encoder')

    print("mode dir", model_load_direction)

    if vis_lossland_scape:
        from vis_loss_landscape import visualize_landscape
        visualize_landscape(model_load_direction, model_var_attack, var_main_encoder, config,
                            raw_dataset.eval_data, config["epsilon"], 300, xent_loss, dataset_type)

    elif black_attack:
        test_black_attack(model_load_direction, model_var_attack, n_Anum_correct, var_main_encoder, attack_steps,
                   attack_step_size, config, dataset_type, raw_dataset, n_mask, layer_values_A['x4'],
                   layer_values_A['pre_softmax'], loss_func,
                   rand_start, use_rand=use_rand, momentum=momentum, load_filename=save_filename)
    else:
        test_model(model_load_direction, model_var_attack, n_Anum_correct, var_main_encoder, attack_steps,
                   attack_step_size, config, dataset_type, raw_dataset, n_mask, layer_values_A['x4'],
                   layer_values_A['pre_softmax'], loss_func,
                   rand_start, use_rand=use_rand, momentum=momentum, save_filename=save_filename)

    print("mode dir", model_load_direction, 'rand start', rand_start, 'loss_func', loss_func, 'step num',attack_steps,
          'step size', attack_step_size, save_filename)


if __name__ == '__main__':
    one_test('cifar10',
             'path/to/the/saved/model',
              7, 2,
              loss_func='cw', rand_start=1)




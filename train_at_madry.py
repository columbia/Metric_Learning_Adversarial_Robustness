import json
import os
import shutil
import socket
import argparse
from datetime import datetime
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

import dataloader.cifar10_input
import dataloader.mnist_input
import dataloader.imagenet_input_new
from learning.model_vanilla import ModelVani
# from learning.model_mnist import ModelMNIST, ModelMNISTBN
from learning.model_mnist_large import ModelMNIST

from pgd_attack import LinfPGDAttack

# from learning.model_imagenet_res import ModelImagenet
# from learning.model_imagenet_wrn import ModelImagenet
# from learning.model_imagenet_wrn_small import ModelImagenet
from learning.model_imagenet_res50 import ModelImagenet

from utils import trainable_in, remove_duplicate_node_from_list, triplet_loss_dict, visualize_imgs, mse_loss, \
    reshape_cal_len, compute_vector_dist_toall_except_own
from learning.eval_within_train import eval_in_train_vanilla
from utils import include_patterns

import matplotlib.pyplot as plt
import pickle

slim = tf.contrib.slim


# Parse input parameters
parser = argparse.ArgumentParser(description='Train Triplet')
parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='dataset to use')
parser.add_argument('--model-dir-postfix', dest='model_dir_postfix', type=str, default='', help='postfix added to directory holding the log')
parser.add_argument('--nat', action='store_true', help='use only natural training')
parser.add_argument('--train-size', dest='train_size', type=int, default=None, help='')

args = parser.parse_args()

config = None
raw_dataset = None
raw_dataset2 = None
model = None
cla_raw_cifar = None


dataset_type = args.dataset
print("using dataset", dataset_type)


assert dataset_type in ['mnist', 'cifar10', 'drebin', 'imagenet', 'cifar100']
model_dir_postfix = args.model_dir_postfix + str(args.train_size) + '_'
# Load in config files and set up parameters
if dataset_type == 'cifar10':
    with open('config_cifar.json') as config_file:
        config = json.load(config_file)
elif dataset_type == 'cifar100':
    with open('config_cifar100.json') as config_file:
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

precision_level = config['precision_level']
precision = tf.float32
if precision_level == 32:
    precision = tf.float32
elif precision_level == 16:
    precision = tf.float16
elif precision_level == 64:
    precision = tf.float64

match_l2 = config["match_l2"]
mul_num = config['mul_num']

label_smoothing = config['label_smoothing']
reuse_emb = config['reuse_embedding'] #TODO:

max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
# step_size_schedule_adam_img_v2 = config['step_size_schedule_adam']
step_size_schedule_finetune = config['step_size_schedule_finetune']

# Adv_classifier_step_size_schedule = config['Adv_classifier_step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
# lambda_match = config['lambda_match']
nat_noise_level = config['nat_noise_level']

train_flag_adv_only = config['train_flag_adv_only']
warming_up = config['warming_up']

is_finetune = config['finetuning']
optimizer = config['optimizer']

regularize_lambda = config['regularize_lambda']
gen_loss_type = config['gen_loss_type']
strong_attack_config = config['strong_attack']

model_dir = config['model_dir_other']
model_load_dir = config['model_load_dir_other']

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

gpu_options = tf.GPUOptions(allow_growth=True)


# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])


# Change path according to host
if socket.gethostname() == 'deep':
    model_dir = config['model_dir']
    model_load_dir = config['model_load_dir']



input_shape = None

# Setting up the data and the model
if dataset_type == 'cifar10':
    input_shape = [None, 32, 32, 3]
    raw_dataset = dataloader.cifar10_input.CIFAR10Data(data_path)
    if config['model_type'] == 'Res20':
        from learning.model_cifar10_resnet import CifarResNet
        model = CifarResNet(precision=precision, ratio=config['mask_ratio'], mode='20')
    elif config['model_type'] == 'Res50':
        from learning.model_cifar10_resnet import CifarResNet
        model = CifarResNet(precision=precision, ratio=config['mask_ratio'], mode='50')
    elif config['model_type'] == 'Res101':
        from learning.model_cifar10_res101 import CifarResNetUpdate
        model = CifarResNetUpdate(precision=precision, ratio=config['mask_ratio'], mode='101')
    elif config['model_type'] == 'VGG':
        from learning.model_cifar_vgg import CifarVGG
        model = CifarVGG(precision=precision, ratio=config['mask_ratio'])
    elif config['model_type'] == 'ConvNet':
        from learning.convnet_cifar import CifarConvNet
        model = CifarConvNet(precision=precision, ratio=config['mask_ratio'])
    else:
        model = ModelVani(precision=precision, ratio=config['mask_ratio'])

elif dataset_type == 'cifar100':
    input_shape = [None, 32, 32, 3]
    raw_dataset = dataloader.cifar10_input.CIFAR100_Data(data_path)
    model = ModelVani(precision=precision, ratio=config['mask_ratio'])

elif dataset_type == 'mnist':
    input_shape = [None, 28, 28]
    train_size = args.train_size
    test_size = None
    reprocess = True

    raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type, train_size=train_size,
                                                   test_size=test_size)
    if config['model_type']=='MLP':
        from learning.model_mnist_mlp import ModelMNISTMLP
        model = ModelMNISTMLP(precision=precision, ratio=config['mask_ratio'])

    else:
        model = ModelMNIST(precision=precision, ratio=config['mask_ratio'])
elif dataset_type == 'drebin':
    input_shape = [None, 545334]
    raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)
    model = ModelDrebin(precision=precision)
elif dataset_type == 'imagenet':
    input_shape = [None, 64, 64, 3]
    raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)
    if config['model_type'] == 'Res20':
        from learning.model_imagenet_res20 import ModelImagenet
        model = ModelImagenet(batch_size=batch_size, precision=precision, label_smoothing=label_smoothing)
    elif config['model_type'] == 'Res50':
        from learning.model_imagenet_res50 import ModelImagenet
        model = ModelImagenet(batch_size=batch_size, precision=precision, label_smoothing=label_smoothing)
    else:
        raise ('error')


global_step = tf.train.get_or_create_global_step()

with tf.variable_scope('input'):
    x_Anat = tf.placeholder(precision, shape=input_shape)
    x_Bnat = tf.placeholder(precision, shape=input_shape)
    y_Ainput = tf.placeholder(tf.int64, shape=None)
    is_training = tf.placeholder(tf.bool, shape=None)

### Get feature vectors
# A
layer_values_A, n_Axent, n_Amean_xent, n_Aweight_decay_loss, n_Anum_correct, n_Aaccuracy, n_Apredict, _ = \
    model._encoder(x_Anat, y_Ainput, is_training)


f_x4_nat = reshape_cal_len(layer_values_A['x4'])[0]

model_var = n_Anum_correct, n_Axent, x_Anat, y_Ainput, is_training, n_Apredict

model_var_attack = x_Anat, n_Axent, y_Ainput, is_training, n_Aaccuracy

saver = tf.train.Saver(max_to_keep=10)
var_main_encoder = trainable_in('main_encoder')

print("finish build up model")


if is_finetune:
    print('finetuning')
    var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
    restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)

    saver_restore = tf.train.Saver(restore_var_list)
    step_size_schedule = step_size_schedule_finetune


### Caculate losses
with tf.variable_scope('train/m_encoder_momentum'):
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)


    total_loss = n_Amean_xent
    total_loss += weight_decay * n_Aweight_decay_loss

    encoder_opt = None
    if optimizer == "SGD":
        encoder_opt = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif optimizer == "adam":
        encoder_opt = tf.train.AdamOptimizer(learning_rate)

    grads1 = encoder_opt.compute_gradients(total_loss, var_list=var_main_encoder)

    train_step_m_encoder = encoder_opt.apply_gradients(grads1)


new_global_step = tf.add(global_step, 1, name='global_step/add')
increment_global_step_op = tf.assign(
    global_step,
    new_global_step,
    name='global_step/assign'
)


attack = LinfPGDAttack(model_var_attack,
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'],
                       dataset_type,
                       attack_suc_ratio=config['attack_suc_ratio'],
                       max_multi=config['max_multi']
                       )

attack_test = LinfPGDAttack(model_var_attack,
                       config['epsilon'],
                       strong_attack_config[0], #num steps
                       strong_attack_config[1], #step size
                       config['random_start'],
                       config['loss_func'],
                       dataset_type,
                       attack_suc_ratio=config['attack_suc_ratio'],
                       max_multi=config['max_multi']
                       )

FGSM = LinfPGDAttack(model_var_attack,
                       config['epsilon'],
                       1,
                       config['epsilon'],
                       config['random_start'],
                       'xent',
                       dataset_type
                       )  # TODO: without momentum

tf.summary.scalar('train_batch_nat accuracy', n_Aaccuracy)
tf.summary.scalar('train_batch_nat xent', n_Axent / batch_size)
tf.summary.scalar('lr', learning_rate)
merged_summaries = tf.summary.merge_all()

# To avoid folder name conflict, append index in the end.
model_dir += model_dir_postfix
new_model_dir = model_dir
postfix_ind = 0
while os.path.exists(new_model_dir):
    new_model_dir = model_dir + '_' + str(postfix_ind)
    postfix_ind += 1

shutil.copytree('.', os.path.join(new_model_dir), ignore=include_patterns('*.py', '*.json'))
model_dir = new_model_dir

fp = open(os.path.join(model_dir, 'log.txt'), 'a')

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:  #

  cifar = None
  cifar_aux = None
  cifar_emb = None
  if dataset_type == 'cifar10' or dataset_type == 'cifar100':
      cifar = dataloader.cifar10_input.AugmentedCIFAR10Data(raw_dataset, sess, model)
      # cifar_emb = dataloader.cifar10_input.AugmentedCIFAR10Data(raw_dataset2, sess, model)
      # cifar_aux = dataloader.cifar10_input.AugmentedCIFAR10Data(cla_raw_cifar, sess, '')

      # cifar = raw_dataset
  elif dataset_type == 'mnist':
      cifar = raw_dataset
  elif dataset_type == 'drebin':
      cifar = raw_dataset
  elif dataset_type == 'imagenet':
      cifar = dataloader.imagenet_input_new.AugmentedIMAGENETData(raw_dataset, sess)
      # cifar = raw_dataset

  eval_dir = os.path.join(model_dir, 'eval')
  # test_summary_writer = tf.summary.FileWriter(eval_dir+'_nat')

  test_summary_writer = tf.summary.FileWriter(eval_dir + '_train_7PGD')
  test_summary_writer_20PGD = tf.summary.FileWriter(eval_dir + '_20PGD')
  test_summary_writer_FGSM = tf.summary.FileWriter(eval_dir + '_FGSM')

  sess.run(tf.global_variables_initializer())

  if is_finetune:
      model_dir_load = tf.train.latest_checkpoint(model_load_dir)
      saver_restore.restore(sess, model_dir_load)


  training_time = 0.0
  best_adv_acc = -1

  cnt = 0
  num_steps, step_size, weight_gen_value = None, None, None


  for ii in range(max_num_training_steps):
    start = timer()
    x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)

    if args.nat:
        x_batch_adv = x_batch
        train_nat_dict_True = {x_Anat: x_batch.astype(np.float32),
                               y_Ainput: y_batch,
                               is_training: True}
    else:
        x_batch_adv = attack.perturb(x_batch, y_batch, True, sess)

        t1 = timer()
        train_adv_dict_True = {x_Anat: x_batch_adv.astype(np.float32),
                               y_Ainput: y_batch,
                               is_training: True}

    train_adv_dict_False = {x_Anat: x_batch_adv.astype(np.float32),
                            y_Ainput: y_batch,
                            is_training: False}
    train_nat_dict_False = {x_Anat: x_batch.astype(np.float32),
                            y_Ainput: y_batch,
                            is_training: False}
    if ii % num_output_steps == 0:
        nat_acc, nat_xent_value = sess.run([n_Aaccuracy, n_Amean_xent], feed_dict=train_nat_dict_False)
        adv_acc, adv_xent_value = sess.run([n_Aaccuracy, n_Amean_xent], feed_dict=train_adv_dict_False)

        str1 = 'Step {}:    ({})\n'.format(ii, datetime.now()) \
               + 'training nat batch accuracy {:.4}%\n'.format(nat_acc * 100) \
               + 'training nat xent {:.4}\n'.format(nat_xent_value) \
               + 'training adv batch accuracy {:.4}%\n'.format(adv_acc * 100) \
               + 'training adv xent {:.4}\n'.format(adv_xent_value) \
            # fp.write(str1)
        print(str1)

    t2 = timer()
    if ii % num_summary_steps == 0:
        summary = sess.run(merged_summaries, feed_dict=train_adv_dict_False)
        test_summary_writer.add_summary(summary, global_step.eval(sess))

    if (ii + 1) % num_checkpoint_steps == 0 and args.nat:
        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)

    if (ii+1) % num_checkpoint_steps == 0 and not args.nat:

        print('-' * 10, 'FGSM', '-' * 10)
        _ = eval_in_train_vanilla(config, model_var, raw_dataset, sess, global_step, test_summary_writer_FGSM,
                                  False, fp, dataset_type, attack_test=FGSM)
        print('-' * 10, '20PGD', '-' * 10)
        adv_acc = eval_in_train_vanilla(config, model_var, raw_dataset, sess, global_step, test_summary_writer_20PGD,
                                        False, fp, dataset_type, attack_test=attack_test)

        print("model dir", model_dir)
        if adv_acc > best_adv_acc or args.nat:
            best_adv_acc = adv_acc

        saver.save(sess,
                   os.path.join(model_dir, 'checkpoint'),
                   global_step=global_step)

    if args.nat:
        sess.run([increment_global_step_op, train_step_m_encoder], feed_dict=train_nat_dict_True)
    else:
        sess.run([increment_global_step_op, train_step_m_encoder], feed_dict=train_adv_dict_True)

    end = timer()

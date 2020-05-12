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
from learning.model_mnist_large import ModelMNIST
from learning.model_imagenet_res50 import ModelImagenet

from pgd_attack import LinfPGDAttack, TargetedGen
from utils import trainable_in, remove_duplicate_node_from_list, triplet_loss_dict, visualize_imgs, mse_loss, \
    reshape_cal_len, compute_vector_dist_toall_except_own
from learning.eval_within_train import eval_in_train_vanilla
from utils import include_patterns

slim = tf.contrib.slim


# Parse input parameters
parser = argparse.ArgumentParser(description='Train Triplet')
parser.add_argument('--dataset', dest='dataset', type=str, default='mnist', help='dataset to use')
parser.add_argument('--gpu', dest='gpu', type=int, default=1, help='num of gpus to use')
parser.add_argument('--model-dir-postfix', dest='model_dir_postfix', type=str, default='', help='postfix added to directory holding the log')
parser.add_argument('--train-size', dest='train_size', type=int, default=None, help='')
parser.add_argument('--random_negative', action='store_true', help='use random negative')
parser.add_argument('--diff_neg', action='store_true', help='use random negative')
parser.add_argument('--bn_tryifbug', action='store_true', help='use random negative')
parser.add_argument('--use_lipschitz', action='store_true', help='use Lipschitz Constant')
# parser.add_argument('--margin_mul', dest='margin_mul', type=int, default=1, help='')


args = parser.parse_args()

assert not (args.diff_neg and args.random_negative)

config = None
raw_dataset = None
raw_dataset2 = None
model = None
cla_raw_cifar = None


dataset_type = args.dataset
print("using dataset", dataset_type)


assert dataset_type in ['mnist', 'cifar10', 'drebin', 'imagenet', 'cifar100']
model_dir_postfix = args.model_dir_postfix
# Load in config files and set up parameters
if dataset_type == 'cifar10':
    with open('config_cifar.json') as config_file:
        config = json.load(config_file)
elif dataset_type == 'mnist':
    with open('config_mnist.json') as config_file:
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

# TBD: set these as input parameters
Use_A_Ap_B = config["Use_A_Ap_B"]
Use_A1_Ap_B = config["Use_A1_Ap_B"]
assert Use_A_Ap_B or Use_A1_Ap_B
A1_Ap_B_num = config["A1_Ap_B_num"]
assert 1 <= A1_Ap_B_num <= 5
Use_B_Bp_A = config["Use_B_Bp_A"]
triplet_loss_layers = config["triplet_loss_layers"] # ['x0', 'x1', 'x2', 'x3', 'x4', 'pre_softmax']
assert len(triplet_loss_layers) > 0

# margin = config["margin"]
margin_list = config["margin_list"]
lamda_triplet = config["lamda_triplet"]
match_l2 = config["match_l2"]
mul_num = config['mul_num']

nat_lam = config["nat_lam"]
margin_mul = config["margin_mul"]
print('margin_mul', margin_mul)

# Note: current config set these parameters to be the same across layers. However, they can be customized for different layers.
triplet_loss_margins = {layer_name:{'A_Ap_B':margin * margin_mul, 'A1_Ap_B_list':[margin for _ in range(A1_Ap_B_num)], 'B_Bp_A':margin} for layer_name, margin in zip(triplet_loss_layers, margin_list)}

loss_coeffs = {'A':nat_lam/(A1_Ap_B_num+2), 'B':nat_lam/(A1_Ap_B_num+2),
               'A1_list':[nat_lam/(A1_Ap_B_num+2) for _ in range(A1_Ap_B_num)],
               'Ap':1, 'Bp':1, 'A_Ap_B':lamda_triplet,
               'A1_Ap_B_list':[lamda_triplet for _ in range(A1_Ap_B_num)], 'B_Bp_A':lamda_triplet}

add_noise_to_X = True if dataset_type != 'drebin' else False

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
strong_attack_config = config['strong_attack']

matchLayerNum = config['matchLayerNum']
triplet_loss_type = config['triplet_loss_type']

# 'triplet_loss_margin_JSdiv' and 'triplet_loss_margin_l2' are not used
triplet_loss_margin_JSdiv = config['triplet_loss_margin_JSdiv']
triplet_loss_margin_l2 = config['triplet_loss_margin_l2']

for layer_name in triplet_loss_margins:
    for k in triplet_loss_margins[layer_name]:
        if triplet_loss_type == 'xent_after_softmax':
            triplet_loss_margins[layer_name][k] = triplet_loss_margin_JSdiv
        elif triplet_loss_type == 'l2':
            triplet_loss_margins[layer_name][k] = triplet_loss_margin_l2

regularize_lambda = config['regularize_lambda']
gen_loss_type = config['gen_loss_type']


model_dir = config['model_dir_other']
model_load_dir = config['model_load_dir_other']

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)

gpu_options = tf.GPUOptions(allow_growth=True)
switch_an_neg = config["switch_a_n"]

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])

label_smoothing = config['label_smoothing']
# Change path according to host
if socket.gethostname() == 'deep':
    model_dir = config['model_dir']
    model_load_dir = config['model_load_dir']


### create folder name according to parameters.
# TBD: consider more parameters and maybe simplify the current one.
sub_folder_name = dataset_type+','
if Use_A_Ap_B:
    sub_folder_name += 'A_Ap_B'+','
if Use_A1_Ap_B:
    sub_folder_name += 'A1_Ap_B_'+str(A1_Ap_B_num)+','
if Use_B_Bp_A:
    sub_folder_name += 'B_Bp_A'
sub_folder_name += '_' + str(args.train_size)
model_dir = os.path.join(model_dir, sub_folder_name)


input_shape = None
### cast_to_int seems not to be used???
cast_to_int = None
# Setting up the data and the model
if dataset_type == 'cifar10':
    input_shape = [None, 32, 32, 3]
    raw_dataset = dataloader.cifar10_input.CIFAR10Data(data_path)
    raw_dataset2 = dataloader.cifar10_input.CIFAR10Data(data_path)
    cla_raw_cifar = dataloader.cifar10_input.SepClaCIFAR10(data_path)
    if config['model_type'] == 'Res50':
        from learning.model_cifar10_resnet import CifarResNet
        model = CifarResNet(precision=precision, ratio=config['mask_ratio'], mode='50')
    else:
        model = ModelVani(precision=precision, ratio=config['mask_ratio'], label_smoothing=label_smoothing)
    cast_to_int = True
elif dataset_type == 'mnist':
    train_size = args.train_size
    test_size = None
    reprocess = True
    raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type, train_size=train_size,
                                                   test_size=test_size)
    raw_dataset2 = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type, train_size=train_size,
                                                    test_size=test_size)
    cla_raw_cifar = dataloader.mnist_input.MNISTDataClassed(data_path, dataset=dataset_type, train_size=train_size,
                                                            test_size=test_size, reprocess=reprocess)

    if config['model_type'] == 'MLP':
        from learning.model_mnist_mlp import ModelMNISTMLP
        model = ModelMNISTMLP(precision=precision, ratio=config['mask_ratio'])
    else:
        model = ModelMNIST(precision=precision, ratio=config['mask_ratio'], label_smoothing=label_smoothing)
    cast_to_int = False
elif dataset_type == 'imagenet':
    input_shape = [None, 64, 64, 3]
    raw_dataset = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)
    raw_dataset2 = dataloader.mnist_input.MNISTData(data_path, dataset=dataset_type)
    cla_raw_cifar = dataloader.mnist_input.MNISTDataClassed(data_path, dataset=dataset_type)
    # model = ModelImagenet(batch_size=batch_size, precision=precision, label_smoothing = label_smoothing)
    cast_to_int = True
    from learning.model_imagenet_res50 import ModelImagenet
    model = ModelImagenet(batch_size=batch_size, precision=precision, label_smoothing=label_smoothing)

global_step = tf.train.get_or_create_global_step()


x_Anat_d_list = []
A1_Ap_B_list = []

with tf.variable_scope('input'):

    x_Anat = tf.placeholder(precision, shape=input_shape)
    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            x_Anat_d_list.append(tf.placeholder(precision, shape=input_shape))
    x_Aadv = tf.placeholder(precision, shape=input_shape)
    x_Aadv_attack = tf.placeholder(precision, shape=input_shape)

    x_Bnat = tf.placeholder(precision, shape=input_shape)
    x_Badv = None
    if Use_B_Bp_A:
        x_Badv = tf.placeholder(precision, shape=input_shape)

    y_Ainput = tf.placeholder(tf.int64, shape=None)
    y_Binput = tf.placeholder(tf.int64, shape=None)
    is_training = tf.placeholder(tf.bool, shape=None)

### Get feature vectors
# A
layer_values_A, n_Axent, n_Amean_xent, _, n_Anum_correct, n_Aaccuracy, n_Apredict, _ = \
    model._encoder(x_Anat, y_Ainput, is_training)
# print("layer_values_A", layer_values_A)
# A'
layer_values_Ap, a_Axent, a_Amean_xent, a_Aweight_decay_loss, a_Anum_correct, a_Aaccuracy, a_Apredict, a_Amask = \
    model._encoder(x_Aadv, y_Ainput, is_training, mask_effective_attack=config['mask_effective_attack'])

# B
layer_values_B, n_Bxent, n_Bmean_xent, _, n_Bnum_correct, n_Baccuracy, _, _ = \
    model._encoder(x_Bnat, y_Binput, is_training)

# B'
if Use_B_Bp_A:
    layer_values_Bp, a_Bxent, a_Bmean_xent, a_Bweight_decay_loss, a_Bnum_correct, a_Baccuracy, a_Bpredict, a_Bmask = \
    model._encoder(x_Badv, y_Binput, is_training)

# A1, A2, ...
if Use_A1_Ap_B:
    for i in range(A1_Ap_B_num):
        layer_values_A1, _, n_Amean_xent_d, _, _, n_Aaccuracy_d, _, _ = \
        model._encoder(x_Anat_d_list[i], y_Ainput, is_training)
        A1_Ap_B_list.append({'layer_values_A1':layer_values_A1, 'n_Amean_xent_d':n_Amean_xent_d, 'n_Aaccuracy_d':n_Aaccuracy_d})



### Calculate triplet loss
triplet_loss_data_A_Ap_B = dict()
mse_loss_A_Ap = dict()
mse_loss_A_B = dict()
triplet_loss_data_A1_Ap_B_list = [dict() for _ in range(A1_Ap_B_num)]
# mse_loss_data_A1_Ap_B_list = [dict() for _ in range(A1_Ap_B_num)]

triplet_loss_data_B_Bp_A = dict()

#construct embed
f_x4_nat=reshape_cal_len(layer_values_A['x4'])[0]

for layer_name in triplet_loss_layers:
    if Use_A_Ap_B:
        if switch_an_neg:
            anchor = layer_values_A[layer_name]  # TODO: already switched anchor and pos
            pos = layer_values_Ap[layer_name]
        else:
            pos = layer_values_A[layer_name]  #TODO: already switched anchor and pos
            anchor = layer_values_Ap[layer_name]
        if Use_B_Bp_A:
            neg = layer_values_Bp[layer_name]
        else:
            neg = layer_values_B[layer_name]

        triplet_loss_data_A_Ap_B[layer_name] = triplet_loss_dict(anchor, pos, neg, triplet_loss_type, regularize_lambda,
                                                                 triplet_loss_margins[layer_name]['A_Ap_B'])
        mse_loss_A_Ap[layer_name] = mse_loss(anchor, pos)
        mse_loss_A_B[layer_name] = mse_loss(anchor, neg)
        # We only add mse here to add the ALP.
    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            if switch_an_neg:
                anchor = A1_Ap_B_list[i]['layer_values_A1'][layer_name]  # TODO: I've changed the a and p here
                pos = layer_values_Ap[layer_name]
            else:
                pos = A1_Ap_B_list[i]['layer_values_A1'][layer_name]  #TODO: I've changed the a and p here
                anchor = layer_values_Ap[layer_name]
            neg = layer_values_B[layer_name]

            triplet_loss_data_A1_Ap_B_list[i][layer_name] = triplet_loss_dict(anchor, pos, neg, triplet_loss_type,
                                                regularize_lambda, triplet_loss_margins[layer_name]['A1_Ap_B_list'][i])
            # mse_loss_data_A1_Ap_B_list[i][layer_name] = mse_loss(anchor, pos)

    if Use_B_Bp_A:
        pos = layer_values_B[layer_name]
        anchor = layer_values_Bp[layer_name]
        neg = layer_values_Ap[layer_name]

        triplet_loss_data_B_Bp_A[layer_name] = triplet_loss_dict(anchor, pos, neg, triplet_loss_type, regularize_lambda, triplet_loss_margins[layer_name]['B_Bp_A'])


model_var_B_Bp_A = x_Anat, y_Ainput, x_Bnat, layer_values_A['x4'], layer_values_A['pre_softmax'], layer_values_B['x4'], layer_values_B['pre_softmax'], is_training
model_var_attack = x_Aadv, a_Axent, y_Ainput, is_training, a_Aaccuracy
# model_var = n_Anum_correct, n_Axent, a_Anum_correct, a_Axent, x_Anat, x_Aadv, y_Ainput, is_training

model_var = n_Anum_correct, n_Axent, x_Anat, y_Ainput, is_training, n_Apredict

saver = tf.train.Saver(max_to_keep=3)
var_main_encoder = trainable_in('main_encoder')

if is_finetune:
    print('finetuning')
    if dataset_type == 'imagenet':
        # restore_var_list = slim.get_variables_to_restore(exclude=tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='logits'))
        var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
        restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)
    else:
        var_main_encoder_var = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES, scope='main_encoder')
        restore_var_list = remove_duplicate_node_from_list(var_main_encoder, var_main_encoder_var)

    saver_restore = tf.train.Saver(restore_var_list)
    step_size_schedule = step_size_schedule_finetune

print("finish build up model")
# print("lambda match", lambda_match)


### Caculate losses
with tf.variable_scope('train/m_encoder_momentum'):
    boundaries = [int(sss[0]) for sss in step_size_schedule]
    boundaries = boundaries[1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(
        tf.cast(global_step, tf.int32),
        boundaries,
        values)


    total_loss = 0
    total_loss += weight_decay * a_Aweight_decay_loss
    total_loss += loss_coeffs['Ap'] * a_Amean_xent

    for layer_name in triplet_loss_layers:
        if Use_A_Ap_B:
            total_loss += (loss_coeffs['A_Ap_B'] * triplet_loss_data_A_Ap_B[layer_name]['triplet_loss'] +
                           match_l2 * mse_loss_A_Ap[layer_name])
        if Use_A1_Ap_B:
            for i in range(A1_Ap_B_num):
                total_loss += loss_coeffs['A1_Ap_B_list'][i] * triplet_loss_data_A1_Ap_B_list[i][layer_name]['triplet_loss']
        if Use_B_Bp_A:
            total_loss += loss_coeffs['Bp'] * a_Bmean_xent
            total_loss += loss_coeffs['B_Bp_A'] * triplet_loss_data_B_Bp_A[layer_name]['triplet_loss']

    # Set coefficients associated to natural values to 0
    if train_flag_adv_only:
        loss_coeffs['A'] = 0
        loss_coeffs['B'] = 0
        for i in range(A1_Ap_B_num):
            loss_coeffs['A1_list'][i] = 0
    else:
        total_loss += loss_coeffs['A'] * n_Amean_xent
        total_loss += loss_coeffs['B'] * n_Bmean_xent

    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            total_loss += loss_coeffs['A1_list'][i] * A1_Ap_B_list[i]['n_Amean_xent_d']

    encoder_opt = None
    if optimizer == "SGD":
        encoder_opt = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif optimizer == "adam":
        encoder_opt = tf.train.AdamOptimizer(learning_rate)
    else:
        raise("NO Optimizer")

    grads1 = encoder_opt.compute_gradients(total_loss, var_list=var_main_encoder)
    train_step_m_encoder = encoder_opt.apply_gradients(grads1)


new_global_step = tf.add(global_step, 1, name='global_step/add')
increment_global_step_op = tf.assign(
    global_step,
    new_global_step,
    name='global_step/assign'
)

from pgd_attack_GPU import LinfPGDAttackGPUImg
model_VarList = x_Aadv, x_Aadv_attack, y_Ainput, is_training
attack_gpu = LinfPGDAttackGPUImg(model_VarList, model, config['epsilon'], config['num_steps'], config['step_size'],
                                 config['random_start'], dataset_type, config)


attack_mild = LinfPGDAttack(model_var_attack,
                            config['epsilon'],
                            config['num_steps'],
                            config['step_size'],
                            config['random_start'],
                            config['loss_func'],
                            dataset_type
                            )  # TODO: without momentum

attack_strong = LinfPGDAttack(model_var_attack,
                              config['epsilon'],
                              strong_attack_config[0],
                              strong_attack_config[1],
                              config['random_start'],
                       'xent',
                              dataset_type
                              )  # TODO: without momentum
#
FGSM = LinfPGDAttack(model_var_attack,
                       config['epsilon'],
                       1,
                       config['epsilon'],
                       config['random_start'],
                       'xent',
                       dataset_type
                       )  # TODO: without momentum

if Use_B_Bp_A:
    targeted_gen = TargetedGen(
        model_var_B_Bp_A,
        config['random_start'],
        gen_loss_type,
        dataset_type
    )
    tf.summary.scalar('pred gen accuracy', a_Baccuracy)

tf.summary.scalar('pred_adv accuracy', a_Aaccuracy)
tf.summary.scalar('pred_nat accuracy', n_Aaccuracy)
tf.summary.scalar('pred_adv xent', a_Axent / batch_size)
tf.summary.scalar('pred_nat xent', n_Axent / batch_size)
if Use_A_Ap_B:
    tf.summary.scalar('A Ap MSE last', mse_loss_A_Ap[triplet_loss_layers[0]])
    tf.summary.scalar('pos neg MSE ratio', mse_loss_A_Ap[triplet_loss_layers[0]] / mse_loss_A_B[triplet_loss_layers[0]])

for layer_name in triplet_loss_layers:
    if Use_A_Ap_B:
        tf.summary.scalar('Triplet loss A', triplet_loss_data_A_Ap_B[layer_name]['triplet_loss'])
    if Use_B_Bp_A:
        tf.summary.scalar('Triplet loss B', triplet_loss_data_B_Bp_A[layer_name]['triplet_loss'])
    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            tf.summary.scalar('Triplet loss nat A 1 with index '+str(i), triplet_loss_data_A1_Ap_B_list[i][layer_name]['triplet_loss'])

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
  if dataset_type == 'cifar10':
      cifar = dataloader.cifar10_input.AugmentedCIFAR10Data(raw_dataset, sess, model)
      cifar_emb = dataloader.cifar10_input.AugmentedCIFAR10Data(raw_dataset2, sess, model)
      cifar_aux = dataloader.cifar10_input.AugmentedCIFAR10Data(cla_raw_cifar, sess, '')

  elif dataset_type == 'mnist':
      cifar = raw_dataset
      cifar_emb = raw_dataset2
      cifar_aux = cla_raw_cifar

  elif dataset_type == 'imagenet':
      cifar = dataloader.imagenet_input_new.AugmentedIMAGENETData(raw_dataset, sess)
      cifar_emb = dataloader.imagenet_input_new.AugmentedIMAGENETData(raw_dataset2, sess)
      cifar_aux = dataloader.imagenet_input_new.AugmentedIMAGENETData(cla_raw_cifar, sess)


  train_advdir = os.path.join(model_dir, 'train/adv')
  if not os.path.exists(train_advdir):
      os.makedirs(train_advdir)
  eval_dir = os.path.join(model_dir, 'eval')
  image_dir = os.path.join(model_dir, 'imgs')
  if not os.path.exists(image_dir):
      os.makedirs(image_dir)

  summary_writer_adv = tf.summary.FileWriter(train_advdir, sess.graph)

  test_summary_writer = tf.summary.FileWriter(eval_dir+'_7PGD')
  test_summary_writer_20PGD = tf.summary.FileWriter(eval_dir + '_20PGD')
  test_summary_writer_FGSM = tf.summary.FileWriter(eval_dir + '_FGSM')
  test_summary_writer_adv = tf.summary.FileWriter(eval_dir + '/adv')

  test_summary_writer_list = [test_summary_writer, test_summary_writer_adv]

  sess.run(tf.global_variables_initializer())

  print("model dir", model_dir)

  if is_finetune:
      model_dir_load = tf.train.latest_checkpoint(model_load_dir)
      saver_restore.restore(sess, model_dir_load)
      print('finetuning', model_dir_load)

  training_time = 0.0
  best_adv_acc = -1

  cnt = 0
  num_steps, step_size, weight_gen_value = None, None, None
  # num_steps, step_size, weight_gen_value = 8, 2, 0.05

  num_k = int(batch_size * mul_num)
  print('select k randomly', num_k)

  aux_mode_flag = False

  cnt_t = 0
  mini_time = 0
  all_time = 0
  for ii in range(max_num_training_steps):

    if args.random_negative:
        neg_image, neg_iamge_label = cifar_emb.train_data.get_next_batch(batch_size, multiple_passes=True)
        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)
        x_batch_adv, f_x4_adv_eval = attack_gpu.perturb(x_batch, y_batch, aux_mode_flag, sess)

    elif args.diff_neg:
        assert mul_num<1

        x_batch_d, y_batch = cifar_emb.train_data.get_next_batch(batch_size,
                                                                 multiple_passes=True)
        A_dict = {x_Anat: x_batch_d.astype(np.float32),
                  y_Ainput: y_batch,
                  is_training: aux_mode_flag}

        f_x4_nat_eval = sess.run(f_x4_nat, feed_dict=A_dict)
        emb_dict = {'x4': f_x4_nat_eval, 'raw': x_batch_d, 'label': y_batch}

        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)
        x_batch_adv, f_x4_adv_eval = attack_gpu.perturb(x_batch, y_batch, aux_mode_flag, sess)
        data_dict = {'x4': f_x4_adv_eval, 'label': y_batch}


        score = compute_vector_dist_toall_except_own(emb_dict, data_dict)

        from utils import get_k_mask
        k_mask = get_k_mask(score, num_k)

        score = k_mask * score
        most_similar_neg = np.argmax(score, axis=1)
        neg_image = emb_dict['raw'][most_similar_neg]
        neg_iamge_label = emb_dict['label'][most_similar_neg]

    else:
        t_start = timer()
        if reuse_emb and ii%mul_num==0 or not reuse_emb or mul_num<1:
            x_batch_d, y_batch = cifar_emb.train_data.get_next_batch(int(batch_size * mul_num),
                                                                   multiple_passes=True)

            A_dict = {x_Anat: x_batch_d.astype(np.float32),
                      y_Ainput: y_batch,
                      is_training: aux_mode_flag}
            f_x4_nat_eval = sess.run(f_x4_nat, feed_dict=A_dict)
            emb_dict = {'x4': f_x4_nat_eval, 'raw':x_batch_d, 'label': y_batch}
            # Construct a sampling of the embedding matrix

        # Compute Adversarial Perturbations
        # Get A'
        # t0 = timer()
        t_minibatch_neg = timer()

        x_batch, y_batch = cifar.train_data.get_next_batch(batch_size,
                                                           multiple_passes=True)

        print('x batch type', x_batch.dtype)
        # t1 = timer()
        # x_batch_adv = attack.perturb(x_batch, y_batch, True, sess)
        assert x_batch.dtype != 'uint8'
        x_batch_adv, f_x4_adv_eval = attack_gpu.perturb(x_batch, y_batch, aux_mode_flag, sess)

        debug=True
        if debug:
            print('diff', np.max(np.abs(x_batch_adv - x_batch)))



        data_dict = {'x4': f_x4_adv_eval, 'label': y_batch}
        # t3 = timer()

        #calculate the most neg sample within this sample
        score = compute_vector_dist_toall_except_own(emb_dict, data_dict)
        most_similar_neg = np.argmax(score, axis=1)
        neg_image = emb_dict['raw'][most_similar_neg]
        neg_iamge_label = emb_dict['label'][most_similar_neg]

    # t4 = timer()
    x_batch_A1_list = []
    imgs_to_visualize = [x_batch, x_batch_adv]

    # Get B
    imgs_to_visualize.append(neg_image)

    # Get B'
    if Use_B_Bp_A:
        x_batch_neg = targeted_gen.perturb(neg_image, x_batch, y_batch, num_steps, step_size, weight_gen_value,
                                           step_size * 2, aux_mode_flag, sess)
        imgs_to_visualize.append(x_batch_neg)

    # Get A1
    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            x_batch_3, y_batch_3 = cifar_aux.train_data.get_next_data_basedon_class(y_batch)
            x_batch_A1_list.append(x_batch_3)
            imgs_to_visualize.append(x_batch_3)
            # print(ii, "A1, sum={:.19f}".format(np.sum(x_batch_3)), "class=", y_batch_3)

    if add_noise_to_X:
        temp = np.random.uniform(-nat_noise_level, nat_noise_level, x_batch.shape)
        x_batch = np.clip(np.rint(x_batch + temp), 0, 255)

    # from utils import visualize_imgs
    # visualize_imgs('/home/mcz/AdvPlot/', imgs_to_visualize, img_ind=ii) #TODO: only for debug

    # print('clean max', np.amax(x_batch))
    # print('adv max', np.amax(x_batch_adv))

    train_mix_dict = {x_Anat: x_batch.astype(np.float32),
                      x_Aadv: x_batch_adv.astype(np.float32),
                      y_Ainput: y_batch,
                      x_Bnat: neg_image.astype(np.float32),
                      y_Binput: neg_iamge_label,
                      is_training: True}

    train_mix_dict_eval = {x_Anat: x_batch.astype(np.float32),
                           x_Aadv: x_batch_adv.astype(np.float32),
                           y_Ainput: y_batch,
                           x_Bnat: neg_image.astype(np.float32),
                           y_Binput: neg_iamge_label,
                           is_training: aux_mode_flag}

    if Use_A1_Ap_B:
        for i in range(A1_Ap_B_num):
            train_mix_dict[x_Anat_d_list[i]] = x_batch_A1_list[i].astype(np.float32)
            train_mix_dict_eval[x_Anat_d_list[i]] = x_batch_A1_list[i].astype(np.float32)
    if Use_B_Bp_A:
        train_mix_dict[x_Badv] = x_batch_neg.astype(np.float32)
        train_mix_dict_eval[x_Badv] = x_batch_neg.astype(np.float32)

    # t5 = timer()


    if ii % num_output_steps == 0:

        ### Run session to get values
        values_to_run = [n_Aaccuracy, a_Aaccuracy, n_Amean_xent, a_Amean_xent]

        if Use_B_Bp_A:
            values_to_run.append(a_Baccuracy)

        for layer_name in triplet_loss_layers:
            if Use_B_Bp_A:
                values_to_run.extend([triplet_loss_data_B_Bp_A[layer_name]['triplet_loss'],
                                      triplet_loss_data_B_Bp_A[layer_name]['pos_dist'],
                                      triplet_loss_data_B_Bp_A[layer_name]['neg_dist'],
                                      triplet_loss_data_B_Bp_A[layer_name]['norm']])

            if Use_A_Ap_B:
                values_to_run.extend([triplet_loss_data_A_Ap_B[layer_name]['triplet_loss'],
                                      triplet_loss_data_A_Ap_B[layer_name]['pos_dist'],
                                      triplet_loss_data_A_Ap_B[layer_name]['neg_dist'],
                                      triplet_loss_data_A_Ap_B[layer_name]['norm']])

            if Use_A1_Ap_B:
                for i in range(A1_Ap_B_num):
                    values_to_run.extend([triplet_loss_data_A1_Ap_B_list[i][layer_name]['triplet_loss'],
                                          triplet_loss_data_A1_Ap_B_list[i][layer_name]['pos_dist'],
                                          triplet_loss_data_A1_Ap_B_list[i][layer_name]['neg_dist'],
                                          triplet_loss_data_A1_Ap_B_list[i][layer_name]['norm']])
            if Use_A_Ap_B:
                values_to_run.extend([mse_loss_A_Ap[layer_name]])

        # # For finding the cause of randomness
        # values_to_run.extend([layer_values_A['pre_softmax'], layer_values_Ap['pre_softmax'], layer_values_B['pre_softmax']])

        values = sess.run(values_to_run, feed_dict=train_mix_dict_eval)

        ### Print related values

        nat_acc, adv_acc, nat_xent_value, adv_xent_value = values[0:4]

        str1 = 'Step {}:    ({})\n'.format(ii, datetime.now()) \
               + 'training nat accuracy {:.4}%\n'.format(nat_acc * 100) \
               + 'training adv accuracy {:.4}%\n'.format(adv_acc * 100) \
               + 'training nat xent {:.4}\n'.format(nat_xent_value) \
               + 'training adv xent {:.4}\n'.format(adv_xent_value)
        if Use_B_Bp_A:
            a_Baccuracy_value = sess.run(a_Baccuracy, feed_dict=train_mix_dict_eval)
            str1 += 'Bp adv accuracy {:.4}%\n'.format(a_Baccuracy_value * 100)


        type_name_list = []
        for layer_name in triplet_loss_layers:
            if Use_A_Ap_B:
                type_name_list.append(layer_name+'_A_Ap_B')
            if Use_B_Bp_A:
                type_name_list.append(layer_name+'_B_Bp_A')
            if Use_A1_Ap_B:
                for i in range(A1_Ap_B_num):
                    type_name_list.append(layer_name+'_A1_Ap_B_'+str(i))

        start_ind = 4
        if Use_B_Bp_A:
            start_ind += 1

        j = 0
        for i in range(start_ind, start_ind+len(type_name_list)*4, 4):
            str1 += type_name_list[j]+'\n' \
            + 'Cos Dist {:.9}\n'.format(values[i]) \
            + 'positive pair dist {:.9}\n'.format(values[i+1]) \
            + 'negative pair dist {:.9}\n'.format(values[i+2]) \
            + 'norm: {:.9}\n'.format(values[i+3])
            j += 1

        if Use_A_Ap_B:
            str1 += 'mse of A and Ap {:.9}\n'.format(values[-1])

        fp.write(str1)
        print(str1)
        #####

    # t6 = timer()

    if ii % num_summary_steps == 0:
        summary = sess.run(merged_summaries, feed_dict=train_mix_dict_eval)
        summary_writer_adv.add_summary(summary, global_step.eval(sess))

    if ii % num_checkpoint_steps == 0 and ii > 0 or is_finetune and ii == 0:
        print('-' * 10, 'FGSM', '-' * 10)
        _ = eval_in_train_vanilla(config, model_var, raw_dataset, sess, global_step, test_summary_writer_FGSM,
                                  False, fp, dataset_type, attack_test=FGSM)
        # print('-' * 10, '7PGD', '-' * 10)
        # _ = eval_in_train_vanilla(config, model_var, raw_dataset, sess, global_step, test_summary_writer_list[0],
        #                           False, fp, dataset_type, attack_test=attack_mild)
        print('-' * 10, '20PGD', '-' * 10)
        adv_acc = eval_in_train_vanilla(config, model_var, raw_dataset, sess, global_step, test_summary_writer_20PGD,
                                        False, fp, dataset_type, attack_test=attack_strong)

        print("model dir", model_dir)
        try:
            print("avg mini time", mini_time * 1.0 / cnt_t, 'avg all time', all_time * 1.0 / cnt_t)
        except:
            pass
        if adv_acc > best_adv_acc:
            best_adv_acc = adv_acc
            saver.save(sess,
                       os.path.join(model_dir, 'checkpoint'),
                       global_step=global_step)
        if dataset_type != 'drebin':
            visualize_imgs(image_dir, imgs_to_visualize, ii)

    sess.run([increment_global_step_op, train_step_m_encoder], feed_dict=train_mix_dict)
    t_finish = timer()

    cnt_t += 1
    mini_time += (t_minibatch_neg - t_start)
    all_time += (t_finish - t_start)




print("avg mini time", mini_time * 1.0 / cnt_t, 'avg all time', all_time * 1.0 / cnt_t)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from learning.model_vanilla import ModelVani
from utils import match_loss, triplet_loss_adversarial
from time import time
# from utils_folder.save_drebin import project_to_manifest
from utils import reshape_cal_len
import dataloader.cifar10_input


class LinfPGDAttackGPUImg:
    def __init__(self, model_VarList, model, epsilon, num_steps, step_size, random_start, dataset_type, config):

        self.x_Aadv, self.x_Aadv_attack, self.y_Ainput, self.is_training = model_VarList
        self.random_start = random_start
        self.epsilon = epsilon
        if dataset_type == 'cifar10' or dataset_type == 'cifar100' or dataset_type == 'imagenet':
            self.upper=255
        else:
            self.upper=1

        x_max = tf.clip_by_value(self.x_Aadv + epsilon, 0, self.upper)
        x_min = tf.clip_by_value(self.x_Aadv - epsilon, 0, self.upper)
        print("the attack upper", self.upper, "the epsilon", epsilon, "step size", step_size)

        new_adv_x = self.x_Aadv_attack

        for loop in range(num_steps):
            _, a_Axent_temp, _, _, _, _, _, _ = \
                                    model._encoder(new_adv_x, self.y_Ainput, self.is_training,
                                                   mask_effective_attack=config['mask_effective_attack'])
            fsm_grad = tf.sign(tf.gradients(a_Axent_temp, new_adv_x)[0])
            new_adv_x = new_adv_x + fsm_grad * step_size
            new_adv_x = tf.clip_by_value(new_adv_x, x_min, x_max)
            new_adv_x = tf.stop_gradient(new_adv_x)

        self.final_new_adv_attack = new_adv_x
        layer_values_Ap_emb, _, _, _, _, _, _, _ = \
            model._encoder(self.final_new_adv_attack, self.y_Ainput, self.is_training,
                           mask_effective_attack=config['mask_effective_attack'])

        self.Ap_emb = reshape_cal_len(layer_values_Ap_emb['x4'])[0]

    def perturb(self, x_batch, y_batch, mode, sess):

        x_batch_att = np.copy(x_batch)
        if self.random_start:
            x_batch_att = x_batch_att.astype('float64') + np.random.uniform(-self.epsilon, self.epsilon,
                                                                            x_batch_att.shape)
            x_batch_att = np.clip(x_batch_att, 0, self.upper)
        gen_att_dict = {
            self.x_Aadv_attack: x_batch_att.astype(np.float32),
            self.x_Aadv: x_batch.astype(np.float32),
            self.y_Ainput: y_batch,
            self.is_training: mode
        }
        x_batch_adv, f_x4_adv_eval = sess.run([self.final_new_adv_attack, self.Ap_emb], feed_dict=gen_att_dict)
        return x_batch_adv, f_x4_adv_eval





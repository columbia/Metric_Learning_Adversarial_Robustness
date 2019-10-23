import math

import numpy as np
import tensorflow as tf


def eval_in_train_vanilla(config, model_varlist, dataset, sess, global_step, summary_writer, is_train, fp, dataset_type, test_natural=True, test_adversary=True, attack_test=None):

    def run_graph(x_batch, total_xent, total_corr, num_pos_examples, total_corr_pos):
        train_mix_dict = {x_Anat: x_batch.astype(np.float32),
                          y_Ainput: y_batch,
                          is_training: is_train}
        cur_corr, cur_xent, y_pred = sess.run([n_Anum_correct, n_Axent, n_Apredict], feed_dict=train_mix_dict)

        if dataset_type == 'drebin':
            num_pos_examples += np.sum(y_batch == 1)
            total_corr_pos += np.sum(y_batch[y_pred == y_batch] == 1)

        total_xent += cur_xent
        total_corr += cur_corr

        return total_xent, total_corr, num_pos_examples, total_corr_pos

    def calculate_stats(total_xent, total_corr, total_corr_pos, num_pos_examples, mode='Natural'):
        avg_xent = total_xent / num_eval_examples
        acc = total_corr / num_eval_examples
        if dataset_type == 'drebin':
            TPR = total_corr_pos / num_pos_examples
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Eval Xent '+mode, simple_value=avg_xent),
            tf.Summary.Value(tag='Eval Accuracy '+mode, simple_value=acc)
        ])
        summary_writer.add_summary(summary, global_step.eval(sess))

        str = '\n'+mode+': {:.2f}%'.format(100 * acc) + '\navg loss: {:.4f}'.format(avg_xent)

        if dataset_type == 'drebin':
            str += '\n'+mode+' True Positive Rate: {:.2f}%, Number of Positive Examples: {:d}'.format(100 * TPR, num_pos_examples)

        return acc, str


    if test_adversary:
        assert attack_test != None

    debug=True
    num_eval_examples = config['num_eval_examples']
    if debug:
        num_eval_examples = 500
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    n_Anum_correct, n_Axent, x_Anat, y_Ainput, is_training, n_Apredict = model_varlist


    total_corr_nat = 0
    total_xent_nat = 0.
    total_corr_pos_nat = 0
    num_pos_examples_nat = 0

    total_corr_adv = 0
    total_xent_adv = 0.
    total_corr_pos_adv = 0
    num_pos_examples_adv = 0

    for ibatch in range(num_batches):
        bstart = ibatch * eval_batch_size
        bend = min(bstart + eval_batch_size, num_eval_examples)

        x_batch = dataset.eval_data.xs[bstart:bend, :]
        y_batch = dataset.eval_data.ys[bstart:bend]

        if dataset_type == 'drebin':
            x_batch = x_batch.toarray()

        if test_natural:
            total_xent_nat, total_corr_nat, num_pos_examples_nat, total_corr_pos_nat = run_graph(x_batch, total_xent_nat, total_corr_nat, num_pos_examples_nat, total_corr_pos_nat)

        if test_adversary:
            x_batch_adv = attack_test.perturb(x_batch, y_batch, is_train, sess)

            total_xent_adv, total_corr_adv, num_pos_examples_adv, total_corr_pos_adv = run_graph(x_batch_adv, total_xent_adv, total_corr_adv, num_pos_examples_adv, total_corr_pos_adv)


        # if ibatch % 10 == 0:
        #     print(eval_batch_size)
        #     if test_natural:
        #         print("Correctly classified natural examples: {}".format(cur_corr_nat))
        #     if test_adversary:
        #         print("Correctly classified adversarial examples: {}".format(cur_corr_adv))


    num_batches = (ibatch + 1)
    returned_acc = None
    str1 = '***TEST***\n'

    if test_natural:
        returned_acc, str = calculate_stats(total_xent_nat, total_corr_nat, total_corr_pos_nat, num_pos_examples_nat, mode='Natural')
        str1 += str
    if test_adversary:
        returned_acc, str = calculate_stats(total_xent_adv, total_corr_adv, total_corr_pos_adv, num_pos_examples_adv, mode='Adversarial')
        str1 += str

    fp.write(str1)
    print(str1)

    return returned_acc




def eval_in_train_multiGPUs(config, model_varlist, dataset, sess, global_step, summary_writer_list,
                            merged_f_summaries, mode, fp, dataType='cifar10'):
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_nat, x_adv, y_input, is_training, n_num_correct, n_xent, a_num_correct, a_xent, adv_x_attacked_list_cat = model_varlist

    if len(summary_writer_list) == 2:
        scalar_w, summary_mix = summary_writer_list
    else:
        scalar_w = summary_writer_list[0]

    total_corr_nat = 0
    total_corr_adv = 0
    total_xent_nat = 0.
    total_xent_adv = 0.
    total_matchloss = 0.

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = dataset.eval_data.xs[bstart:bend, :]
      y_batch = dataset.eval_data.ys[bstart:bend]

      if config['data_path'] == 'Drebin_data':
          x_batch = x_batch.toarray()

      gen_dict = {
          x_adv:x_batch.astype(np.float32),
          y_input: y_batch,
          is_training: mode
      }
      x_batch_adv = sess.run(adv_x_attacked_list_cat, feed_dict=gen_dict)

      train_mix_dict = {x_nat: x_batch.astype(np.float32),
                        x_adv: x_batch_adv.astype(np.float32),
                        y_input: y_batch,
                        is_training: mode}

      cur_corr_nat, cur_xent_nat, cur_corr_adv, cur_xent_adv = sess.run(
          [n_num_correct, n_xent, a_num_correct, a_xent],
          feed_dict=train_mix_dict)

      total_xent_nat += cur_xent_nat
      total_xent_adv += cur_xent_adv
      total_corr_nat += cur_corr_nat
      total_corr_adv += cur_corr_adv
      # total_matchloss += matchLoss

      if ibatch % 10 == 0:
          print(eval_batch_size)
          print("Correctly classified natural examples: {}".format(cur_corr_nat))
          print("Correctly classified adversarial examples: {}".format(cur_corr_adv))
          #TODO: delete break


    num_batches = (ibatch + 1)
    avg_xent_nat = total_xent_nat / num_eval_examples
    avg_xent_adv = total_xent_adv / num_eval_examples
    avg_matchloss = total_matchloss / num_batches
    acc_nat = total_corr_nat / num_eval_examples
    acc_adv = total_corr_adv / num_eval_examples

    summary = tf.Summary(value=[
        tf.Summary.Value(tag='eval xent adv', simple_value=avg_xent_adv),
        tf.Summary.Value(tag='eval xent nat', simple_value=avg_xent_nat),
        tf.Summary.Value(tag='eval accuracy adv', simple_value=acc_adv),
        tf.Summary.Value(tag='eval accuracy nat', simple_value=acc_nat)
    ])
    scalar_w.add_summary(summary, global_step.eval(sess))

    print('***TEST**')
    print('natural: {:.2f}%'.format(100 * acc_nat))
    print('adversarial: {:.2f}%'.format(100 * acc_adv))
    print('avg nat loss: {:.4f}'.format(avg_xent_nat))
    print('avg adv loss: {:.4f}'.format(avg_xent_adv))
    print('avg matchloss: {:.9f}'.format(avg_matchloss))
    # print('last matchloss: {:.9f}'.format(matchLoss))
    print("*****")

    str1 = '***TEST**\n' + ('natural: {:.2f}%'.format(100 * acc_nat)) + '\nadversarial: {:.2f}%'.format(100 * acc_adv) + \
           '\navg nat loss: {:.4f}'.format(avg_xent_nat) + \
           '\navg adv loss: {:.4f}'.format(avg_xent_adv) + '\navg matchloss: {:.9f}\n'.format(avg_matchloss)
    # '\nlast matchloss: {:.9f}'.format(matchLoss)
    fp.write(str1)
    return acc_adv

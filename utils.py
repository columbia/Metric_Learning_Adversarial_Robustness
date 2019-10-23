import tensorflow as tf
import os

from fnmatch import filter
from os.path import isdir, join
import matplotlib.pyplot as plt
import numpy as np


def visualize_imgs(save_path, img_list, img_ind):
    x_batch = img_list[0]
    batch_size = x_batch.shape[0]

    vis_img = None
    denominator = 1
    edge = x_batch[0].shape[0]
    step = edge + 3

    if edge == 32:
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list), 3))
        denominator = 255
    elif edge == 28:
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list)))
    # Warning: Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers). May need to be fixed.
    elif edge == 64:
        print(x_batch[0])
        vis_img = np.zeros(((step) * batch_size, (step) * len(img_list), 3)).astype(np.float32)
        denominator = 1  #now the imagenet range from 0 to 1

    for jj, x in enumerate(img_list):
        for ii in range(batch_size):
            vis_img[(ii) * step:(ii) * step + edge, step*jj:step*jj+edge] = x[ii] * 1.0 / denominator

    # vis_img[(ii) * step:(ii) * step + edge, step*4:step*4 + edge] = (x_batch_2[ii] - x_batch_neg[ii]) / denominator

    # print('max', np.max(np.abs(x_batch_2[ii] - x_batch_neg[ii])))
    dpi = 80
    figsize = vis_img.shape[1] * 1.0 / dpi, vis_img.shape[0] * 1.0 / dpi
    fig = plt.figure(figsize=figsize)

    plt.imshow(vis_img)
    # plt.show()
    fig.savefig(os.path.join(save_path, 'plot_{}.png'.format(img_ind)), dpi=dpi)


def include_patterns(*patterns):
    """Factory function that can be used with copytree() ignore parameter.

    Arguments define a sequence of glob-style patterns
    that are used to specify what files to NOT ignore.
    Creates and returns a function that determines this for each directory
    in the file hierarchy rooted at the source directory when used with
    shutil.copytree().
    """
    def _ignore_patterns(path, names):
        keep = set(name for pattern in patterns
                            for name in filter(names, pattern))
        ignore = set(name for name in names
                        if name not in keep and not isdir(join(path, name)))
        return ignore
    return _ignore_patterns


def trainable_in(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

def select_grad(grads, name):
    for each in grads:
        if name in each[1].name:
            return each[0]


def remove_duplicate_node_from_list(A, B):
    result = A
    for EB in B:
        flag=True
        for EA in A:
            if EB == EA:
                # print('find duplicate', EA)
                flag=False
                break
        if flag:
            result.append(EB)
    return result


def reshape_cal_len(x):
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
        prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    return x, prod_non_batch_dimensions


def l2_norm(a):
    a_norm = tf.reduce_sum(a * a + 1e-20, axis=1) ** 0.5
    b = a / tf.expand_dims(a_norm, 1)
    return b

def l2_norm_reshape(a):
    a, len = reshape_cal_len(a)
    a_norm = tf.reduce_sum(a * a + 1e-20, axis=1) ** 0.5
    b = a / tf.expand_dims(a_norm, 1)
    return b


def _relu(x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x)

# def triplet_loss(a, p, n, alpha = 1.0):
#     distance = F.sum((a - p) ** 2.0, axis=1) - F.sum((a - n) ** 2.0, axis=1) + alpha
#     return F.average(F.relu(distance)) / 2

def triplet_loss(a, p, n, loss_type, regularize=0, margin=0.0):
    a, len = reshape_cal_len(a)
    p, len = reshape_cal_len(p)
    n, len = reshape_cal_len(n)
    positive_dist, negative_dist = None, None

    if 'cos' == loss_type:
        norm_a = l2_norm(a)
        norm_p = l2_norm(p)
        norm_n = l2_norm(n)

        positive_dist = 1 - tf.abs(tf.reduce_sum(norm_a * norm_p, axis=1))
        negative_dist = 1 - tf.abs(tf.reduce_sum(norm_a * norm_n, axis=1))

    elif loss_type == 'l2':
        positive_dist = tf.reduce_sum((a - p)**2, axis=1) #TODO: possible bug in axis
        negative_dist = tf.reduce_sum((a - n)**2, axis=1)

    elif loss_type == 'xent_after_softmax':
        print("using loss xent_after_softmax")
        epsilon = 1e-8
        a_soft = tf.nn.softmax(a)
        p_soft = tf.nn.softmax(p)
        n_soft = tf.nn.softmax(n)

        positive_dist = 0.5 * tf.reduce_sum(a_soft * tf.log(a_soft / (p_soft + epsilon))
                                                           + p_soft * tf.log(p_soft / (a_soft + epsilon)), axis=1)
        negative_dist = 0.5 * tf.reduce_sum(a_soft * tf.log(a_soft / (n_soft + epsilon))
                                                           + n_soft * tf.log(n_soft / (a_soft + epsilon)), axis=1)

    # tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    # temp = positive_dist - negative_dist + margin
    #
    # triplet_loss = tf.reduce_mean(tf.maximum(positive_dist - negative_dist + margin, 0.0), axis=0)
    triplet_loss = tf.reduce_mean(tf.maximum(positive_dist - negative_dist + margin, 0.0), axis=0)
    positive_dist = tf.reduce_mean(positive_dist, axis=0)
    negative_dist = tf.reduce_mean(negative_dist, axis=0)

    norm = tf.reduce_mean(a * a + p * p + n * n)
    triplet_loss = triplet_loss + regularize * norm

    return triplet_loss, positive_dist, negative_dist, norm


def triplet_loss_dict(a, p, n, loss_type, regularize=0, margin=0.0):
    triplet_loss_v, positive_dist, negative_dist, norm = triplet_loss(a, p, n, loss_type, regularize, margin)
    return {'triplet_loss':triplet_loss_v, 'pos_dist':positive_dist, 'neg_dist':negative_dist, 'norm':norm}


def triplet_loss_adversarial(a, p, n, loss_type, margin=0.9):
    a, len = reshape_cal_len(a)
    p, len = reshape_cal_len(p)
    n, len = reshape_cal_len(n)

    norm_a = l2_norm(a)
    norm_p = l2_norm(p)
    norm_n = l2_norm(n)
    if 'cos' in loss_type:
        positive_dist = 1 - tf.reduce_mean(tf.reduce_sum(norm_a * norm_p, axis=1), axis=0)
        negative_dist = 1 - tf.reduce_mean(tf.reduce_sum(norm_a * norm_n, axis=1), axis=0)

    loss = positive_dist - negative_dist
    return loss


def mse_loss(a, b):
    a_fea, len = reshape_cal_len(a)
    b_fea, len = reshape_cal_len(b)

    return tf.reduce_mean(tf.reduce_sum((a_fea - b_fea) ** 2, axis=1), axis=0)


def mean(a):
    return tf.reduce_mean(a)


def var(a):
    return tf.reduce_mean((a-mean(a)) ** 2)

def match_loss(nat_fea, adv_fea, loss_type, scope=None, epislon=0.01):  # TODO: note that this is not symmetrical

  if loss_type == 'mse':
      a_fea, len = reshape_cal_len(nat_fea)
      b_fea, len = reshape_cal_len(adv_fea)

      return tf.reduce_mean(tf.reduce_sum((a_fea - b_fea) ** 2, axis=1), axis=0)

  if loss_type == 'cos':
      nat_fea, len = reshape_cal_len(nat_fea)
      adv_fea, len = reshape_cal_len(adv_fea)

      norm_nat = l2_norm(nat_fea)
      norm_adv = l2_norm(adv_fea)

      cos_similarity = tf.reduce_mean(tf.reduce_sum(norm_nat * norm_adv, axis=1), axis=0)
      return cos_similarity

  if loss_type == 'l1_cos':

      nat_fea, len = reshape_cal_len(nat_fea)
      adv_fea, len = reshape_cal_len(adv_fea)

      norm_nat = l2_norm(nat_fea)
      norm_adv = l2_norm(adv_fea)

      cos_similarity = tf.reduce_mean(tf.reduce_sum(norm_nat * norm_adv, axis=1), axis=0)
      # return tf.reduce_mean((1 - tf.abs(cos_similarity))**2)
      return _relu(1 - tf.abs(cos_similarity))

  if loss_type == 'l2_cos':

      nat_fea, len = reshape_cal_len(nat_fea)
      adv_fea, len = reshape_cal_len(adv_fea)

      norm_nat = l2_norm(nat_fea)
      norm_adv = l2_norm(adv_fea)

      cos_similarity = tf.reduce_mean(tf.reduce_sum(norm_nat * norm_adv, axis=1), axis=0)
      # return tf.reduce_mean((1 - tf.abs(cos_similarity))**2)
      return (1 - tf.abs(cos_similarity))**2   #TODO:Note, abs is critical here, because both -1 and 1 in NN indicate similarity

  if loss_type == 'l2_cos_rand':
      # norm1 = tf.sqrt(tf.reduce_sum(tf.multiply(fea1, fea1)))
      # norm2 = tf.sqrt(tf.reduce_sum(tf.multiply(fea2, fea2)))
      # return tf.reduce_sum(tf.multiply(fea1, fea2)) / norm1 / norm2

      nat_fea, len = reshape_cal_len(nat_fea)
      adv_fea, len = reshape_cal_len(adv_fea)
      with tf.variable_scope(scope+'gated_noise'):
        w = tf.get_variable(
              'DW', adv_fea.shape,
              initializer=tf.uniform_unit_scaling_initializer(factor=1.0))

        w_norm = tf.nn.l2_normalize(w, axis=[0, 1])  # tf.convert_to_tensor
        w = w / w_norm

        noise_add0 = tf.random.normal(w.shape, mean=0, stddev=epislon)
        noise_add1 = tf.random.normal(w.shape, mean=0, stddev=epislon)

        nat_fea = nat_fea + w * noise_add0
        adv_fea = adv_fea + w * noise_add1


      norm_nat = tf.nn.l2_normalize(nat_fea, 1)
      norm_adv = tf.nn.l2_normalize(adv_fea, 1)

      cos_similarity = tf.losses.cosine_distance(norm_adv, tf.stop_gradient(norm_nat), dim=1)
      return (1 - tf.abs(cos_similarity))**2


def class_into_seperate_np(embed_dict, layer_list_name, class_num):

    cnt_class = np.zeros((class_num), dtype=np.int64)
    for i in range(embed_dict['label'].shape[0]):
        cnt_class[int(embed_dict['label'][i, 0])] += 1

    print("num of each class", cnt_class)

    class_dict = dict()
    for i in range(class_num):
        class_dict[i] = dict()

        for each in layer_list_name:
            class_dict[i][each] = np.zeros((cnt_class[i], embed_dict[each].shape[1]))

    c_class = np.zeros((class_num), dtype=np.int64)
    for ii in range(embed_dict[layer_list_name[0]].shape[0]):
        label_int = int(embed_dict['label'][ii, 0])
        # print("label", label_int)
        for layer_name in layer_list_name:
            class_dict[label_int][layer_name][c_class[label_int], :] = embed_dict[layer_name][ii, :]
        c_class[label_int] += 1
        # print('c class', c_class)

    return class_dict


def compute_vector_dist(embed_dict_class, f_adv_eval, layer_list_name_old, attacked_to_these_class, correct):
    vis_num = 4
    episilon  =1e-7

    batch_size = attacked_to_these_class.shape[0]
    import matplotlib.pyplot as plt

    layer_list_name = layer_list_name_old.copy()
    layer_list_name = layer_list_name[:-1]

    fig = plt.figure()

    cnt=1
    for j, layer_name in enumerate(layer_list_name):
        for ii in range(vis_num):

            fea = f_adv_eval[j][ii, :]
            attacked_to = attacked_to_these_class[ii]
            origin_class = f_adv_eval[-1][ii]
            is_correct = correct[ii]
            print(fea.shape)
            fea = (fea / np.sum(fea**2) **0.5 + episilon)
            fea = np.reshape(fea, (fea.shape[0], 1))

            emb_save = embed_dict_class[int(attacked_to)][layer_name]

            emb_save = emb_save / np.expand_dims((np.sum(emb_save**2, axis=1) ** 0.5 + episilon), 1)

            cos_dist = np.dot(emb_save, fea)

            print('result shape', cos_dist.shape)

            ax = fig.add_subplot(len(layer_list_name),vis_num,  cnt)
            # ax.subplot(all, j+1, ii+1)
            ax.hist(cos_dist[:, 0], 50, density=True)
            ax.set_title(layer_name + "_{}-to-{}-iscor={}".format(origin_class, attacked_to, is_correct))
            cnt += 1
    plt.show()


def compute_vector_dist_toall_except_own_plot(embed_dict_class, f_adv_eval, layer_list_name_old, correct):
    episilon = 1e-7

    layer_list_name = layer_list_name_old.copy()
    layer_list_name = layer_list_name[:-1]

    fig = plt.figure()
    vis_num = min(4, f_adv_eval[-1].shape[0])

    cnt=1

    for j, layer_name in enumerate(layer_list_name):
        fea = f_adv_eval[j]
        # for ii in range(vis_num):
        #     fea = f_adv_eval[j][ii, :]
        #     origin_class = f_adv_eval[-1][ii]

        fea = fea / np.expand_dims((np.sum(fea**2, axis=1) ** 0.5 + episilon), 1) # batchsize * fealen
        embed = embed_dict_class[layer_name]   # total_size * fealen
        embed = embed / np.expand_dims((np.sum(embed ** 2, axis=1) ** 0.5 + episilon), 1)
        embed = np.transpose(embed, (1, 0))  # fealen * total_size

        fea_label = f_adv_eval[-1]  #batch * 1
        fea_label = np.expand_dims(fea_label, 1)
        emb_label = embed_dict_class['label']
        # print("emb_label", emb_label.shape, fea_label.shape)
        emb_label = np.transpose(emb_label, (1, 0))  # 1* batch

        similar_score = np.dot(fea, embed)  ##batch * total_size

        fea_label_replicate = np.repeat(fea_label, emb_label.shape[0], axis=1)
        embed_label_replicate = np.repeat(emb_label, fea_label.shape[0], axis=0)

        mask_same = fea_label_replicate == embed_label_replicate

        similar_score_ofdif_class = (1-mask_same) * similar_score
        # print(fea_label, emb_label)
        # print(similar_score_ofdif_class)

        for ii in range(vis_num):
            ax = fig.add_subplot(len(layer_list_name), vis_num, cnt)
            # ax.subplot(all, j+1, ii+1)
            ax.hist(similar_score_ofdif_class[ii], 50, density=True)
            origin_class = f_adv_eval[-1][ii]
            is_correct = correct[ii]
            ax.set_title(layer_name + "_{}-to-iscor={}".format(origin_class, is_correct))
            cnt += 1
    plt.show()



def compute_vector_dist_toall_except_own(embed_dict_class, data_dict):
    episilon = 1e-7
    fea = data_dict['x4']
    embed = embed_dict_class['x4']  # total_size * fealen

    # if use_mean:
    #     fea = fea - np.expand_dims(np.mean(fea, axis=1), 1)
    #     embed = embed - np.expand_dims(np.mean(embed, axis=1), 1)

    fea = fea / np.expand_dims((np.sum(fea**2, axis=1) ** 0.5 + episilon), 1) # batchsize * fealen #TODO: minus mean
    #TODO: but there's no need to mean 0 the vector to compute the inner product

    embed = embed / np.expand_dims((np.sum(embed ** 2, axis=1) ** 0.5 + episilon), 1)
    embed = np.transpose(embed, (1, 0))  # fealen * total_size

    fea_label = data_dict['label']  #batch * 1
    fea_label = np.expand_dims(fea_label, 1)
    emb_label = embed_dict_class['label']
    emb_label = np.expand_dims(emb_label, 1)
    # print("emb_label", emb_label.shape, fea_label.shape)
    emb_label = np.transpose(emb_label, (1, 0))  # 1* batch

    similar_score = np.dot(fea, embed)  ##batch * total_size

    fea_label_replicate = np.repeat(fea_label, emb_label.shape[0], axis=1)
    embed_label_replicate = np.repeat(emb_label, fea_label.shape[0], axis=0)

    mask_same = fea_label_replicate == embed_label_replicate

    similar_score_ofdif_class = (1-mask_same) * similar_score
    return similar_score_ofdif_class


def get_k_mask(input, num_k):
    ones = np.ones(input.shape[1])
    diag_matrix = np.diag(ones)

    ratio = num_k * 1.0 / input.shape[1]

    mask = np.random.rand(input.shape[0], input.shape[1]) < ratio
    mask = mask.astype(int)
    mask = mask+diag_matrix

    mask = mask>0
    mask = mask.astype(float)
    return mask



def get_embed(layer_list_name, f_nat_eval, embed_dict):
    for ind, layer_name in enumerate(layer_list_name):
        # if fea.ndim == 1:
        #     fea = np.reshape(fea, (fea.shape[0], 1))
        embed_dict[layer_name] = f_nat_eval[ind]
    return embed_dict


def match_mean_var_triplet(a, b, c):
    a_fea, len = reshape_cal_len(a)
    b_fea, len = reshape_cal_len(b)
    c_fea, len = reshape_cal_len(c)

    a_mean = tf.reduce_mean(a_fea, axis=1)
    b_mean = tf.reduce_mean(b_fea, axis=1)
    c_mean = tf.reduce_mean(c_fea, axis=1)

    mean_all = tf.reduce_mean((a_mean - b_mean)**2 + (a_mean - c_mean)**2 + (c_mean - b_mean)**2, axis=0)

    a_var = tf.reduce_mean((a_fea - tf.expand_dims(a_mean, axis=1)) ** 2, axis=1)
    b_var = tf.reduce_mean((b_fea - tf.expand_dims(b_mean, axis=1)) ** 2, axis=1)
    c_var = tf.reduce_mean((c_fea - tf.expand_dims(c_mean, axis=1)) ** 2, axis=1)

    var_all = tf.reduce_mean((a_var - b_var)**2 + (a_var - c_var)**2 + (b_var - c_var)**2, axis=0)

    return mean_all + var_all


def get_adap(ii, margin_step, margin_range, neg_dict_size_step, neg_dict_size_range, batch_size, total_train_num):

    multiplier = total_train_num*1.0 / batch_size

    margin_step *= multiplier
    neg_dict_size_step *= multiplier

    margin_value = -1
    if ii<margin_step:
        margin_value = margin_range[0] + (margin_range[1] - margin_range[0]) * 1.0 / margin_step  * ii
    else:
        margin_value = margin_range[1]

    dict_size=neg_dict_size_range[0]

    if ii < neg_dict_size_step:
        dict_size = neg_dict_size_range[0] + (neg_dict_size_range[1] - neg_dict_size_range[0]) * 1.0 / neg_dict_size_step * ii
    else:
        dict_size = neg_dict_size_range[1]

    return margin_value, dict_size



# def normalize_vec(a):
#     a_fea, len = reshape_cal_len(a)
#     a_mean = tf.reduce_mean(a_fea, axis=1)
#     a_temp = a_fea - tf.expand_dims(a_mean, axis=1)
#     a_var = tf.expand_dims(tf.reduce_mean(a_temp ** 2, axis=1), axis=1)
#     return a_temp / ()












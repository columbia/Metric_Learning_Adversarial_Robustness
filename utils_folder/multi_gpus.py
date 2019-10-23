import tensorflow as tf


def tf_sum_scalar(scalar_list):
    for i, each in enumerate(scalar_list):
        scalar_list[i] = tf.expand_dims(each, 0)

    cat_list = tf.concat(scalar_list, 0)
    return tf.reduce_mean(cat_list)


def average_gradients(tower_grads):
    average_grads = []
    # print(tower_grads)
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        cnt = 0
        # print('grad_and_vars', grad_and_vars)
        flag=0

        for g, v in grad_and_vars:  #accumulate same grad in different GPUs
            # Add 0 dimension to the gradients to represent the tower.
            # print("debug", g, cnt)
            # if cnt == 0:
            #     var = v
            # else:
            #     var += v
            cnt += 1
            if g == None:
                flag=1
            else:
                expanded_g = tf.expand_dims(g, 0)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

        if flag==1:
            grad=None
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

# By default, all variables will be placed on '/gpu:0'
# So we need a custom device function, to assign all variables to '/cpu:0'
# Note: If GPUs are peered, '/gpu:0' can be a faster option
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return "" + ps_device
        else:
            return device

    return _assign
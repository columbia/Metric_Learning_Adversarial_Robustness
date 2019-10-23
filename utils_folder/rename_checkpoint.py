"""https://gist.github.com/batzner/7c24802dd9c5e15870b4b56e22135c96"""
import sys, getopt, os

import tensorflow as tf

usage_str = 'python tensorflow_rename_variables.py --checkpoint_dir=path/to/dir/ ' \
            '--replace_from=substr --replace_to=substr --add_prefix=abc --dry_run'

# example:
# python rename_checkpoint.py --checkpoint_dir=models/naturally_trained/ --add_prefix=main_encoder/

def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run):
    print('check dir', checkpoint_dir)
    model_file = tf.train.latest_checkpoint(checkpoint_dir)
    print('model file', model_file)
    # checkpoint = tf.train.get_checkpoint_state(model_file)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(model_file):
            # Load the variable
            var = tf.contrib.framework.load_variable(model_file, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            # saver.save(sess, 'models/model_0_renamed/madry_rename')
            # saver.save(sess, 'models/naturally_trained_rename/model_0')
            saver.save(sess, 'cifar_public/madry_pub_rename')


def main(argv):
    checkpoint_dir = None
    replace_from = None
    replace_to = None
    add_prefix = None
    dry_run = False

    try:
        opts, args = getopt.getopt(argv, 'h', ['help=', 'checkpoint_dir=', 'replace_from=',
                                               'replace_to=', 'add_prefix=', 'dry_run'])
    except getopt.GetoptError:
        print(usage_str)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(usage_str)
            sys.exit()
        elif opt == '--checkpoint_dir':
            checkpoint_dir = arg
        elif opt == '--replace_from':
            replace_from = arg
        elif opt == '--replace_to':
            replace_to = arg
        elif opt == '--add_prefix':
            add_prefix = arg
        elif opt == '--dry_run':
            dry_run = True

    if not checkpoint_dir:
        print('Please specify a checkpoint_dir. Usage:')
        print(usage_str)
        sys.exit(2)

    rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run)


if __name__ == '__main__':
    main(sys.argv[1:])


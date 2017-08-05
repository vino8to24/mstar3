import os
import tensorflow as tf
from mstar import mstar3_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 100,
                            """Number of images to process in a batch.""")
tf.app.flags.DEFINE_string('data_dir',
                           '/media/jiangxin/493CA763B844807F/qingshuisi/data_sets/bin/mstar',
                           """Path to the mstar data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using pf16.""")

m_size = mstar3_input.m_size
num_classes = mstar3_input.num_classes
num_examples_per_epoch_for_train = mstar3_input.num_per_epoch_for_train
num_examples_per_epoch_for_eval = mstar3_input.num_per_epoch_for_eval

moving_average_decay = 0.9999
num_epochs_per_decay = 350
learning_rate_decay_factor = 0.1
initial_learning_rate = 0.1

def variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = variable_on_cpu(name, shape,
                          tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def distorted_inputs():
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = mstar3_input.distorted_inputs(data_dir=data_dir,
                                                   batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels

def inputs(eval_data):
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = FLAGS.data_dir
    images, labels = mstar3_input.inputs(eval_data=eval_data,
                                         data_dir=data_dir,
                                         batch_size=FLAGS.batch_size)
    if FLAGS.use_fp16:
        images = tf.cast(images, tf.float16)
        labels = tf.cast(labels, tf.float16)
    return images, labels


def inference(images):
    # conv1 scope
    with tf.variable_scope('conv1') as scope:
        kernal = variable_with_weight_decay('weights',
                                            shape=[5, 5, 1, 64],
                                            stddev=5e-2,
                                            wd=0.0)
        conv = tf.nn.conv2d(images, kernal, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
    #summary

    #pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    #norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm1')

    #conv2
    with tf.variable_scope('conv2') as scope:
        kernal = variable_with_weight_decay('weights',
                                            shape=[5, 5, 64, 64],
                                            stddev=5e-2,
                                            wd=0.0)
        conv = tf.nn.conv2d(norm1, kernal, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    #norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75, name='norm2')
    #pool2
    pool2=tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                         padding='SAME', name='pool2')

    #local3
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384],
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    #local4
    with tf.variable_scope('local4') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192],
                                             stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    with tf.variable_scope('softmax_linear') as scope:
        weights = variable_with_weight_decay('weights', shape=[192, num_classes],
                                             stddev=1/192.0, wd=0.004)
        biases = variable_on_cpu('biases', [num_classes], tf.constant_initializer(0.0))
        softmax_linear = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)


    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits, labels, name='cross_entropy_per_example'
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def add_loss_summaries(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses+[total_loss])
    for l in losses+[total_loss]:
        tf.scalar_summary(l.op.name + '(raw)', l)
        tf.scalar_summary(l.op.name, loss_averages.average(l))
    return loss_averages_op


def train(total_loss, global_step):
    num_batch_per_epoch = num_examples_per_epoch_for_train / FLAGS.batch_size
    decay_steps = int(num_batch_per_epoch * num_epochs_per_decay)

    lr = tf.train.exponential_decay(initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    learning_rate_decay_factor,
                                    staircase=True)
    tf.scalar_summary('learning_rate', lr)

    loss_averages_op = add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    #apply gradients
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    #track the moving average of all trainable variables
    variable_averages = tf.train.ExponentialMovingAverage(
        moving_average_decay, global_step
    )
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    with tf.control_dependencies([apply_gradient_op, variable_averages_op]):
        train_op = tf.no_op(name='train')
    return train_op
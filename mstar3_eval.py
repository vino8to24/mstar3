import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from mstar import mstar3

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/media/jiangxin/493CA763B844807F/qingshuisi/data_sets/bin/mstar',
                            """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
#tf.app.flags.DEFINE_string('checkpoint_dir', '/media/jiangxin/493CA763B844807F/qingshuisi/data_sets/bin/mstar',
#                          """Directory where to read model checkpoint.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60*5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 588,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                            """Whether to run eval only once.""")

def eval_once(top_k_op):

    # start the queue runners
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count  =0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            # compute precision @1
            precision = true_count / total_sample_count
            print('%s: precision @1 = %.3f' % (datetime.now(), precision))

        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    # Get images and labels from mastar
    eval_data = FLAGS.eval_data == 'test'
    images, labels = mstar3.inputs(eval_data=eval_data)
    # Build a graph that computes the logits predictions from the
    # inference model
    logits = mstar3.inference(images)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    # Calculate predictions
    variable_average = tf.train.ExponentialMovingAverage(
        mstar3.moving_average_decay
    )
    variable_to_restore = variable_average.variables_to_restore()

    while True:
        eval_once(top_k_op)
        if FLAGS.run_once:
            break
        time.sleep(FLAGS.eval_interval_secs)


evaluate()


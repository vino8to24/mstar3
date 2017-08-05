import tensorflow as tf
from mstar import mstar3
from six.moves import xrange
import time
import numpy as np
from datetime import datetime

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir',
                           '/media/jiangxin/493CA763B844807F/qingshuisi/data_sets/bin/mstar',
                           """Directory where to write event logs""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")



global_step = tf.Variable(0, trainable=False)

images, labels = mstar3.distorted_inputs()
#print(images)
#print(labels)

logits = mstar3.inference(images)
#print(logits)
loss = mstar3.loss(logits, labels)

train_op = mstar3.train(loss, global_step)

init = tf.global_variables_initializer()



sess = tf.Session()#config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
sess.run(init)
tf.train.start_queue_runners(sess=sess)

for step in xrange(FLAGS.max_steps):
    start_time = time.time()
    _, loss_value = sess.run([train_op, loss])
    #print('logits: ', sess.run(logits))
    #print('images: ', sess.run(images))
    #print('labels: ', sess.run(labels))
    #print('loss: ', sess.run(loss))
    duration = time.time() - start_time
    #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 10 == 0:
        num_examples_per_step = FLAGS.batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f err_rate = %.2f(%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, loss_value/128*100,
                            examples_per_sec, sec_per_batch))
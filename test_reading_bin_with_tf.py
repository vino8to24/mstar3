import tensorflow as tf


class Mstar3Record():
    pass
result = Mstar3Record()
label_bytes = 1
result.height = 128
result.width = 128
result.depth = 1
image_bytes = result.height * result.width * result.depth
record_bytes = image_bytes + label_bytes

filename = ['mstar3_train.bin']
filename_queue = tf.train.string_input_producer(filename)

reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
result.key, value = reader.read(filename_queue)

record_bytes = tf.decode_raw(value, tf.uint8)
#print(record_bytes)
result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.float32)
depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])
result.float32image = tf.transpose(depth_major, [1, 2, 0])

init = tf.global_variables_initializer()
with tf.Session as sess:
    sess.run(init)



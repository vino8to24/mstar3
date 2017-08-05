import tensorflow as tf
import os

m_size = 64
num_classes = 3
num_per_epoch_for_train = 697
num_per_epoch_for_eval = 588

def read_mstar3(filename_queue):
    class Mstar3Record(object):
        pass
    result = Mstar3Record()

    label_bytes = 1
    result.height = 128
    result.width = 128
    result.depth = 1
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes*4)
    result.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.float32)

    result.label = tf.cast(tf.slice(record_bytes, [0], [label_bytes]), tf.float32)

    #one_hot_label = tf.zeros(shape=[1, 3], dtype=tf.float32)
    #print(one_hot_label)
    #for i in range(3):
    #    one_hot_label[1, int(result.label)-1] = 1
    #result.label = tf.cast(result.label, tf.uint8)
    #print(result.label)
    #one_hot_label = tf.one_hot(result.label, 3)
    #result.label = one_hot_label
    #print(result.label)
    depth_major = tf.reshape(tf.slice(record_bytes, [label_bytes], [image_bytes]),
                             [result.depth, result.height, result.width])

    result.float32image = tf.transpose(depth_major, [1, 2, 0])

    return result


def generate_image_and_label_batch(image, label, min_queue_examples,
                                   batch_size, shuffle):
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size = batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3* batch_size,
            min_after_dequeue=min_queue_examples
        )
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3*batch_size
        )
    # summary

    return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
    filename = [os.path.join(data_dir, 'mstar3_train.bin')]
    if not filename:
        raise ValueError('Failed to find file: ' + filename)

    filename_queue = tf.train.string_input_producer(filename)

    read_input = read_mstar3(filename_queue)
    reshaped_image = tf.cast(read_input.float32image, tf.float32)

    height = m_size
    width = m_size
    # random crop
    #distorted_image = tf.random_crop(reshaped_image, [height, width, 1])
    distorted_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

    # random brightness
    distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)

    # random contrast
    distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

    float_image = tf.image.per_image_standardization(distorted_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_per_epoch_for_train *
                             min_fraction_of_examples_in_queue)
    print('Filling queue with %d mstar images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)
    return generate_image_and_label_batch(float_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          shuffle=True)

def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filename = [os.path.join(data_dir, 'mstar3_train.bin')]
        num_examples_per_epoch = num_per_epoch_for_train
    else:
        filename = [os.path.join(data_dir, 'mstar3_test.bin')]
        num_examples_per_epoch = num_per_epoch_for_eval

    if not filename:
        raise ValueError('Failed to find file: ' + filename)

    filename_queue = tf.train.string_input_producer(filename)

    read_input = read_mstar3(filename_queue)
    reshaped_image = tf.cast(read_input.float32image, tf.float32)

    height = m_size
    width = m_size

    resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, width, height)

    float_image = tf.image.per_image_standardization(resized_image)

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch*min_fraction_of_examples_in_queue)

    return generate_image_and_label_batch(float_image, read_input.label,
                                          min_queue_examples, batch_size,
                                          shuffle=False)
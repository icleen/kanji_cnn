from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import h5py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

import kanji_prepper as prep

import tensorflow as tf
from tensorflow.python import debug as tf_debug

# FLAGS = None

# making the onehot labels for the hiragana data
def onehot_labels(list, classes):
    out = np.zeros(shape=(len(list), classes), dtype=np.int32)
    for i, item in enumerate(list):
        out[i][int(item)] = 1
    return out

# get accuracy
def get_accuracy(validation, v_labels):
    test_batch = 500
    acc = 0.0
    length = int(len(validation) / test_batch)
    for i in range(length):
        a = i*test_batch
        acc += accuracy.eval(feed_dict={
            x: validation[a:a + test_batch],
            y_: v_labels[a:a + test_batch],
            keep_prob: 1.0})
    acc /= length
    return acc

# setting up the debug filter
def has_inf_or_nan(datum, tensor):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))

def main(_):
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    # Hyper-parameters
    width, height = 32, 32
    size = (width, height)
    classes = 1721
    batch_size = 50
    steps = 1000
    save_location = "/tmp/cnn_kanji_simple"

    # Import data
    training, t_labels, validation, v_labels = prep.data_from_base('training_data')
    t_labels = onehot_labels(t_labels, classes)
    v_labels = onehot_labels(v_labels, classes)

    print('data imported')

    sess = tf.InteractiveSession()

    # Create the model
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, width * height])
        y_ = tf.placeholder(tf.float32, [None, classes])

    with tf.name_scope('input_reshape'):
        x_image = tf.reshape(x, [-1,width,height,1])
        tf.summary.image('input', x_image, classes)


    # setting up the cnn
    def weight_variable(shape, nme):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=nme)

    def bias_variable(shape, nme):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=nme)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)

    # adding the first convolutional layer
    with tf.name_scope('conv_layer1'):
        with tf.name_scope('weights'):
            W_conv1 = weight_variable([5, 5, 1, 32], "w1")
            variable_summaries(W_conv1)
        with tf.name_scope('biases'):
            b_conv1 = bias_variable([32], "b1")
            variable_summaries(b_conv1)
        with tf.name_scope('activation'):
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            tf.summary.histogram('activations', h_conv1)

    # adding the first pooling layer
    with tf.name_scope('pooling1'):
        h_pool1 = max_pool_2x2(h_conv1)
        pool1_img = tf.reshape(h_pool1, [-1,width,height,1])
        tf.summary.image('pool1', pool1_img, classes)

    # adding the third convolutional layer
    with tf.name_scope('conv_layer2'):
        with tf.name_scope('weights'):
            W_conv3 = weight_variable([5, 5, 32, 64], "w3")
            variable_summaries(W_conv3)
        with tf.name_scope('biases'):
            b_conv3 = bias_variable([64], "b3")
            variable_summaries(b_conv3)
        with tf.name_scope('activation'):
            h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
            tf.summary.histogram('activations', h_conv3)

    # the second pooling layer
    with tf.name_scope('pooling2'):
        h_pool2 = max_pool_2x2(h_conv3)
        pool2_img = tf.reshape(h_pool2, [-1,width,height,1])
        tf.summary.image('pool2', pool2_img, classes)

    # adding the fifth convolutional layer
    with tf.name_scope('conv_layer3'):
        with tf.name_scope('weights'):
            W_conv5 = weight_variable([5, 5, 64, 64], "w5")
            variable_summaries(W_conv5)
        with tf.name_scope('biases'):
            b_conv5 = bias_variable([64], "b5")
            variable_summaries(b_conv5)
        with tf.name_scope('activation'):
            h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
            tf.summary.histogram('activations', h_conv5)

    # the third pooling layer
    h_pool3 = max_pool_2x2(h_conv5)

    #adding the final layer
    with tf.name_scope('final_layer'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([4 * 4 * 64, 1024], "W_fc1")
            variable_summaries(W_fc1)
        with tf.name_scope('biases'):
            b_fc1 = bias_variable([1024], "b_fc1")
            variable_summaries(b_fc1)
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
            tf.summary.histogram('pre_activations', preactivate)
        h_fc1 = tf.nn.relu(preactivate)
        tf.summary.histogram('activations', h_fc1)

    # adding the dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # adding the readout layer
    with tf.name_scope('readout_layer'):
        with tf.name_scope('weights'):
            W_fc3 = weight_variable([1024, classes], "w_read")
            variable_summaries(W_fc3)
        with tf.name_scope('biases'):
            b_fc3 = bias_variable([classes], "b_read")
            variable_summaries(b_fc3)
        with tf.name_scope('Wx_plus_b'):
            y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3
            tf.summary.histogram('activations', y_conv)

    with tf.name_scope('cross_entropy'):
        # The raw formulation of cross-entropy,
        #
        #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
        #                                 reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        # outputs of 'y', and then average across the batch.
        diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
        with tf.name_scope('total'):
            cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Test trained model
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/kanji_simple/logs/kanji_with_summaries/train', sess.graph)
    test_writer = tf.summary.FileWriter('/tmp/tensorflow/kanji_simple/logs/kanji_with_summaries/test')
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()
    if os.path.exists(os.path.join(save_location)):
        saver.restore(sess, save_location + "/model.ckpt")

    epoch = -1
    test_batch = 500
    # Train
    for i in range(steps):
        a = i*batch_size % len(training)
        batchx = training[a:a + batch_size]
        batchy = t_labels[a:a + batch_size]
        summary, _ = sess.run([merged, train_step], feed_dict={x: batchx, y_: batchy, keep_prob: 0.5})
        train_writer.add_summary(summary, i)
        if i%100 == 0:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary, _ = sess.run([merged, train_step],
                                  feed_dict={x: batchx, y_: batchy, keep_prob: 0.5},
                                  options=run_options,
                                  run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            print('Adding run metadata for', i)
            summary, acc = sess.run([merged, accuracy], feed_dict={x: validation,y_: v_labels,keep_prob: 1.0})
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
            save_path = saver.save(sess, save_location + "/model.ckpt")

    save_path = saver.save(sess, save_location + "/model.ckpt")
    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/tmp/tensorflow/kanji_simple/input_data',
        help='Directory for storing input data')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='/tmp/tensorflow/kanji_simple/logs/kanji_with_summaries',
        help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

import kanji_prepper as prep

import tensorflow as tf
from tensorflow.python import debug as tf_debug

FLAGS = None

# making the onehot labels for the hiragana data
def onehot_labels(list, classes):
    out = np.zeros(shape=(len(list), classes), dtype=np.int32)
    for i, item in enumerate(list):
        out[i][int(item)] = 1
    return out

# setting up the debug filter
def has_inf_or_nan(datum, tensor):
    return np.any(np.isnan(tensor)) or np.any(np.isinf(tensor))

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

def main(_):
    # if tf.gfile.Exists(FLAGS.log_dir):
    #     tf.gfile.DeleteRecursively(FLAGS.log_dir)
    # tf.gfile.MakeDirs(FLAGS.log_dir)
    sess = tf.InteractiveSession()

    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)

    # get accuracy
    def get_accuracy(step):
        test_batch = 500
        tot_acc = 0.0
        length = int(len(validation) / test_batch)
        for i in range(length):
            a = i*test_batch
            summary, acc = sess.run([merged, accuracy], feed_dict={
                x: validation[a:a + test_batch],
                y_: v_labels[a:a + test_batch],
                keep_prob: 1.0})
            # res = str('%d_%d'% (step, i))
            # print(res)
            validation_writer.add_summary(summary, step + i)
            tot_acc += acc
        tot_acc /= length
        return tot_acc
    # Hyper-parameters
    width, height = 32, 32
    size = (width, height)
    classes = 1721
    batch_size = 50
    steps = 5000
    learn_rate = 0.0001
    save_location = "/tmp/cnn_kanji_log"

    # Import data
    training, t_labels, validation, v_labels = prep.data_from_base('training_data')
    t_labels = onehot_labels(t_labels, classes)
    v_labels = onehot_labels(v_labels, classes)

    print('data imported')

    # Create the model
    x = tf.placeholder(tf.float32, [None, width * height])
    x_image = tf.reshape(x, [-1,width,height,1])
    y_ = tf.placeholder(tf.float32, [None, classes])

    # adding the first convolutional layer
    with tf.name_scope('conv_layer1'):
        with tf.name_scope('weights'):
            W_conv1 = weight_variable([5, 5, 1, 32], "w1")
            variable_summaries(W_conv1)
        b_conv1 = bias_variable([32], "b1")
        with tf.name_scope('activations'):
            h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
            variable_summaries(h_conv1)

    # adding the second convolutional layer
    with tf.name_scope('conv_layer2'):
        with tf.name_scope('weights'):
            W_conv2 = weight_variable([5, 5, 32, 32], "w2")
            variable_summaries(W_conv2)
        b_conv2 = bias_variable([32], "b2")
        with tf.name_scope('activations'):
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
            variable_summaries(h_conv2)

    # adding the first pooling layer
    with tf.name_scope('pooling1'):
        h_pool1 = max_pool_2x2(h_conv2)
        pool1_img = tf.reshape(h_pool1, [-1,width,height,1])
        tf.summary.image('pool1', pool1_img, classes)


    # adding the third convolutional layer
    with tf.name_scope('conv_layer3'):
        with tf.name_scope('weights'):
            W_conv3 = weight_variable([5, 5, 32, 64], "w3")
            variable_summaries(W_conv3)
        b_conv3 = bias_variable([64], "b3")
        with tf.name_scope('activations'):
            h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
            variable_summaries(h_conv3)

    # adding the fourth convolutional layer
    with tf.name_scope('conv_layer4'):
        with tf.name_scope('weights'):
            W_conv4 = weight_variable([5, 5, 64, 64], "w4")
            variable_summaries(W_conv4)
        b_conv4 = bias_variable([64], "b4")
        with tf.name_scope('activations'):
            h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
            variable_summaries(h_conv4)

    # the second pooling layer
    with tf.name_scope('pooling2'):
        h_pool2 = max_pool_2x2(h_conv4)
        pool2_img = tf.reshape(h_pool2, [-1,width,height,1])
        tf.summary.image('pool2', pool2_img, classes)

    # adding the fifth convolutional layer
    with tf.name_scope('conv_layer5'):
        with tf.name_scope('weights'):
            W_conv5 = weight_variable([5, 5, 64, 64], "w5")
            variable_summaries(W_conv5)
        b_conv5 = bias_variable([64], "b5")
        with tf.name_scope('activations'):
            h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)
            variable_summaries(h_conv5)

    # the third pooling layer
    with tf.name_scope('pooling3'):
        h_pool3 = max_pool_2x2(h_conv5)
        pool3_img = tf.reshape(h_pool3, [-1,width,height,1])
        tf.summary.image('pool3', pool3_img, classes)

    #adding the final layer
    with tf.name_scope('fully_connected1'):
        with tf.name_scope('weights'):
            W_fc1 = weight_variable([4 * 4 * 64, 1024], "W_fc1")
            variable_summaries(W_fc1)
        b_fc1 = bias_variable([1024], "b_fc1")
        h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 64])
        with tf.name_scope('activations'):
            h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
            variable_summaries(h_fc1)

    # adding the dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # adding another fully connected layer
    with tf.name_scope('fully_connected2'):
        with tf.name_scope('weights'):
            W_fc2 = weight_variable([1024, 1024], "W_fc2")
        b_fc2 = bias_variable([1024], "b_fc2")
        with tf.name_scope('activations'):
            h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # adding the readout layer
    with tf.name_scope('readout_layer'):
        with tf.name_scope('weights'):
            W_fc3 = weight_variable([1024, classes], "w_read")
        b_fc3 = bias_variable([classes], "b_read")
        with tf.name_scope('activations'):
            y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    # Test trained model
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Add ops to save and restore all the variables.
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/cnn_kanji/logs/kanji_with_summaries/train', sess.graph)
    validation_writer = tf.summary.FileWriter('/tmp/tensorflow/cnn_kanji/logs/kanji_with_summaries/validation')
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
            train_accuracy = accuracy.eval(feed_dict={
                x:batchx, y_: batchy, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            save_path = saver.save(sess, save_location + "/model.ckpt")
        if a < batch_size:
            epoch += 1
            acc = get_accuracy(i)
            print("epoch %d, validation accuracy %g"%(epoch, acc))

    acc = get_accuracy(steps)
    print("validation accuracy %g"%acc)
    save_path = saver.save(sess, save_location + "/model.ckpt")
    train_writer.close()
    test_writer.close()

if __name__ == '__main__':
    tf.app.run()

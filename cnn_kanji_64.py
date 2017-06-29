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
    # Hyper-parameters
    width, height = 64, 64
    size = (width, height)
    classes = 1721
    batch_size = 50
    steps = 10000
    learn_rate = 0.0001
    save_location = "/tmp/cnn_kanji_64"

    # Import data
    training, t_labels, validation, v_labels = prep.data_from_base('training_data_64')
    t_labels = onehot_labels(t_labels, classes)
    v_labels = onehot_labels(v_labels, classes)

    print('data imported')

    # Create the model
    x = tf.placeholder(tf.float32, [None, width * height])
    x_image = tf.reshape(x, [-1,width,height,1])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, classes])

    # adding the first convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32], "w1")
    b_conv1 = bias_variable([32], "b1")
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # adding the second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 32], "w2")
    b_conv2 = bias_variable([32], "b2")
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

    # adding the first pooling layer
    h_pool1 = max_pool_2x2(h_conv2)

    # adding the third convolutional layer
    W_conv3 = weight_variable([5, 5, 32, 64], "w3")
    b_conv3 = bias_variable([64], "b3")
    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    # adding the fourth convolutional layer
    W_conv4 = weight_variable([5, 5, 64, 64], "w4")
    b_conv4 = bias_variable([64], "b4")
    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)

    # the second pooling layer
    h_pool2 = max_pool_2x2(h_conv4)

    # adding the fifth convolutional layer
    W_conv5 = weight_variable([5, 5, 64, 64], "w5")
    b_conv5 = bias_variable([64], "b5")
    h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    # the second pooling layer
    h_pool3 = max_pool_2x2(h_conv5)

    #adding the final layer
    W_fc1 = weight_variable([8 * 8 * 64, 1024], "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")

    h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # adding the dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # adding another fully connected layer
    # W_fc2 = weight_variable([1024, 2048], "W_fc2")
    # b_fc2 = bias_variable([2048], "b_fc2")
    # h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # adding the readout layer
    W_fc3 = weight_variable([1024, classes], "w_read")
    b_fc3 = bias_variable([classes], "b_read")

    y_conv = tf.matmul(h_fc1_drop, W_fc3) + b_fc3

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    sess.run(tf.global_variables_initializer())
    if os.path.exists(os.path.join(save_location)):
        saver.restore(sess, save_location + "/model.ckpt")

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    epoch = -1
    test_batch = 500
    # Train
    for i in range(steps):
        a = i*batch_size % len(training)
        batchx = training[a:a + batch_size]
        batchy = t_labels[a:a + batch_size]
        train_step.run(feed_dict={x: batchx, y_: batchy, keep_prob: 0.5})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batchx, y_: batchy, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            save_path = saver.save(sess, save_location + "/model.ckpt")
        if a < batch_size:
            epoch += 1
            acc = 0.0
            length = int(len(validation) / test_batch)
            for i in range(length):
                a = i*test_batch
                acc += accuracy.eval(feed_dict={
                    x: validation[a:a + test_batch],
                    y_: v_labels[a:a + test_batch],
                    keep_prob: 1.0})
            acc /= length
            print("epoch %d, test accuracy %g"%(epoch, acc))

    acc = 0.0
    length = int(len(validation) / test_batch)
    for i in range(length):
        a = i*test_batch
        acc += accuracy.eval(feed_dict={
            x: validation[a:a + test_batch],
            y_: v_labels[a:a + test_batch],
            keep_prob: 1.0})
    acc /= length
    print("test accuracy %g"%acc)
    # print("test accuracy %g"%accuracy.eval(feed_dict={
    #     x: validation, y_: v_labels, keep_prob: 1.0}))
    save_path = saver.save(sess, save_location + "/model.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

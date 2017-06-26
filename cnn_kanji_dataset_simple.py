from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import prep_kanji_dataset as prep
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

# making the onehot labels for the hiragana data
def onehot_labels(list, classes):
    out = np.zeros(shape=(len(list), classes), dtype=np.int32)
    for i, item in enumerate(list):
        out[i][int(item)] = 1
    return out

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
    width, height = 32, 32
    size = (width, height)
    classes = 90
    batch_size = 50
    steps = 1000
    save_location = "/tmp/cnn_kanji_dataset_simple"

    # Import data
    file_path = 'kanji_dataset/characters/'
    file_list, folder_list = prep.get_files_list(file_path)
    classes = len(folder_list)
    dictionary = prep.make_dictionary(folder_list)

    train_file_list, val_file_list = prep.split_file(file_list)

    training, t_labels = prep.get_data(train_file_list, dictionary, size)
    t_labels = np.asarray(t_labels, dtype=np.int32)

    validation, v_labels = prep.get_data(val_file_list, dictionary, size)
    v_labels =  np.asarray(v_labels, dtype=np.int32)
    v_labels = onehot_labels(v_labels, classes)

    print('data imported')

    # Create the model
    x = tf.placeholder(tf.float32, [None, width * height])
    W = tf.Variable(tf.zeros([width * height, classes]))
    b = tf.Variable(tf.zeros([classes]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, classes])

    # adding in the convolutional layer
    W_conv1 = weight_variable([5, 5, 1, 32], "w1")
    b_conv1 = bias_variable([32], "b1")
    x_image = tf.reshape(x, [-1,width,height,1])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # adding the second convolutional layer
    W_conv2 = weight_variable([5, 5, 32, 64], "w2")
    b_conv2 = bias_variable([64], "b2")

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # adding the third convolutional layer
    W_conv3 = weight_variable([5, 5, 64, 128], "w3")
    b_conv3 = bias_variable([128], "b3")

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    #adding the final layer
    W_fc1 = weight_variable([4 * 4 * 128, 1024], "w_flat")
    b_fc1 = bias_variable([1024], "b_flat")

    h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

    # adding the dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # adding the readout layer
    W_fc2 = weight_variable([1024, classes], "w_read")
    b_fc2 = bias_variable([classes], "b_read")

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
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    if os.path.exists(os.path.join(save_location)):
        saver.restore(sess, save_location + "/model.ckpt")

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Train
    for i in range(steps):
        a = i*batch_size % len(training)
        batchx = training[a:a + batch_size]
        batchy = t_labels[a:a + batch_size]
        batchy = onehot_labels(batchy, classes)
        train_step.run(feed_dict={x: batchx, y_: batchy, keep_prob: 0.5})
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batchx, y_: batchy, keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("test accuracy %g"%accuracy.eval(feed_dict={
                x: validation, y_: v_labels, keep_prob: 1.0}))
            save_path = saver.save(sess, save_location + "/model.ckpt")

    print("test accuracy %g"%accuracy.eval(feed_dict={
        x: validation, y_: v_labels, keep_prob: 1.0}))
    save_path = saver.save(sess, save_location + "/model.ckpt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

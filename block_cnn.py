from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def variable_summaries(var):
    """ Function for the TensorBoard summaries
    Adds the mean, standard deviation and histogram to
    the TensorBoard summaries

    Args:
        var: tensor of variable to add to summaries
    """
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.histogram('histogram', var)


def conv2d(name, input, ksize, f_in, f_out):
    """ Function for the convolutional layers.
    CNN layers with 1 x 1 stride and zero padding

    Args:
        name: name of the scope
        input: input to the layer
        ksize: filter size
        f_in: number of filters from previous layer (depth of input)
        f_out: number of filters for this layer (depth of output)

    Returns:
        Tensor of the output of the layer
    """
    with tf.variable_scope(name):
        # Weights and bias definition
        with tf.variable_scope('weights'):
            weights = tf.get_variable('weights',
                                shape=[ksize, ksize, f_in, f_out],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(weights)

        with tf.variable_scope('bias'):
            bias = tf.get_variable('bias',
                                shape=[f_out],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(bias)

        # Convolutional layer
        conv = tf.nn.conv2d(input,
                            filter=weights,
                            strides=[1, 1, 1, 1],
                            padding='SAME') + bias

        # Activation layer
        return tf.nn.relu(conv, name=name)


def max_pool(name, input, ksize):
    """ Function for the max pool layer
    Max pool layer with 2 x 2 kernel size, 2 x 2 stride and zero padding.

    Args:
        name: name of scope
        input: input tensor for the max pool layer
        ksize: filter size

    Returns:
        Tensor of the output of the layer
    """
    with tf.variable_scope(name):
        return tf.nn.max_pool(input,
                              ksize=[1, ksize, ksize, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name=name)


def fully_connected(name, input, f_in, f_out, keep_prob=1.0):
    """
    Function for a fully_connected layer
    Arg:
        name: name of scope
        input: input to the layer
        f_in: depth of input
        f_out: number of filters (depth of output)
        keep_prob: keep probability for drop out
    Returns:
        Tensor of the output of the layer
    """
    with tf.variable_scope(name):
        # Weights and bias definition
        with tf.variable_scope('weights'):
            weights = tf.get_variable('weights',
                                      shape=[f_in, f_out],
                                      initializer=tf.truncated_normal_initializer())
            variable_summaries(weights)

        with tf.variable_scope('bias'):
            bias = tf.get_variable('bias', shape=[f_out],
                                   initializer=tf.random_normal_initializer())
            variable_summaries(bias)

        # Fully connected layer operation
        fc = tf.nn.relu(tf.matmul(input, weights) + bias, name=name)

        # Apply dropout
        return tf.nn.dropout(fc, keep_prob, name=name+'_drop')


def inference(image, num_classes, keep_prob):
    """
    Graph definition
    Args:
        image: input image
        num_classes: number of classes
        keep_prob: keep probability for drop out
                    (in fully connected layers)
    Returns:
        Logits from the softmax layer
    """
    # Convolutional layer 1
    conv1 = conv2d('conv1', input=image, ksize=3, f_in=3, f_out=32)

    # Convolutional layer 2
    conv2 = conv2d('conv2', input=conv1, ksize=3, f_in=32, f_out=32)

    # Maxpool 1
    pool1 = max_pool('max_pool1', conv2, 3)

    # Convolutional layer 3
    conv3 = conv2d('conv3', input=pool1, ksize=3, f_in=32, f_out=64)

    # Convolutional layer 4
    conv4 = conv2d('conv4', input=conv3, ksize=3, f_in=64, f_out=64)

    # Maxpool 2
    pool2 = max_pool('max_pool2', conv4, 3)

    # Convolutional layer 5
    conv5 = conv2d('conv5', input=pool2, ksize=3, f_in=64, f_out=128)

    # Convolutional layer 6
    conv6 = conv2d('conv6', input=conv5, ksize=3, f_in=128, f_out=128)

    # Maxpool 3
    pool3 = max_pool('max_pool3', conv6, 3)

    # Convolutional layer 7
    conv7 = conv2d('conv7', input=pool3, ksize=3, f_in=128, f_out=256)

    # Convolutional layer 8
    conv8 = conv2d('conv8', input=conv7, ksize=3, f_in=256, f_out=256)

    # Flatten layer (for fully_connected layer)
    with tf.name_scope('flatten'):
        flat_dim = conv8.get_shape()[1].value * conv8.get_shape()[2].value * conv8.get_shape()[3].value
        flat = tf.reshape(conv8, [-1, flat_dim])

    # Fully connected layer 1
    fc1 = fully_connected('fc1', flat, flat_dim, 1024, keep_prob)

    # Fully connected layer 2
    fc2 = fully_connected('fc2', fc1, 1024, 128, keep_prob)

    # Softmax layer
    with tf.variable_scope('softmax') as scope:
        # Weights and bias definition
        with tf.variable_scope('weights'):
            weights = tf.get_variable('weights',
                                      shape=[128, num_classes],
                                      initializer=tf.truncated_normal_initializer())
            variable_summaries(weights)

        with tf.variable_scope('bias'):
            bias = tf.get_variable('bias', shape=[num_classes],
                                   initializer=tf.random_normal_initializer())
            variable_summaries(bias)

        # Softmax layer
        logits = tf.add(tf.matmul(fc2, weights, name=scope.name), bias)

    return logits


def loss(logits, labels):
    """
    Calculates the loss
    Args:
        logits: logits from the softmax layer
        labels: labels corresponding to the input images
    Returns:
        The mean softmax cross entropy loss
    """
    with tf.name_scope('loss'):
        # Define loss
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                           logits=logits),
            name='loss')
        tf.summary.scalar('loss', loss)

    return loss


def train(loss, learning_rate):
    """
    Train model definition (computes and applies the gradients)
    Args:
        loss: calculated loss from training
        learning_rate: learning rate for RMS prop optimizer
    Return:
        Operation that update the variables
    """
    with tf.name_scope('train'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        train_op = optimizer.minimize(loss, global_step=global_step)

    return train_op


def evaluation(logits, labels):
    """
    Calculates the classification accuracy
    Args:
        logits: logits from the model
        labels: labels corresponding to the input image
    Returns:
         Classification accuracy [0,1)
    """
    with tf.name_scope('evaluate'):
        # Evaluate accuracy
        correct_prediction = tf.nn.in_top_k(logits, labels, 1)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return accuracy



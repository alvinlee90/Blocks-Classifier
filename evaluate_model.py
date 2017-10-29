from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def read_and_decode(filename_queue, img_size):
    """
    Function to read and decode the tfrecord file of the dataset
    Arg:
        filename_queue: list of the tfrecord files
        img_size: size (height/width) of the images (square aspect
        ratio)

    Returns:
        image, label: images and corresponding labels from the dataset
    """
    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image_raw'], tf.float32)

    # Set the shape of the images
    image = tf.reshape(image, [img_size, img_size, 3])

    # Convert to between [-0.5, 0.5)
    image = image * (1.0 / 255) - 0.5

    # Convert label to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label

def inputs(filename, batch_size, img_size):
    """
    Function to
    Arg:
        filename: path to the tfrecord
        batch_size: size of the batch
        num_epochs: number of epochs/iterations
        img_size: size

    Returns:
        images, labels: images and labels for the batch
    """
    with tf.name_scope('input'):
        # Create a list of filenames and pass it to a queue
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=1)

        image, label = read_and_decode(filename_queue, img_size)

        # Creates batches by randomly shuffling tensors
        # Creates batches by randomly shuffling tensors
        images, labels = tf.train.batch([image, label],
                                        batch_size=batch_size,
                                        capacity=100 + 3*batch_size,
                                        num_threads=2,
                                        min_after_dequeue=100)

    return images, labels


def test_model():
    with tf.Graph().as_default():
        images, labels = inputs(filename=FLAGS.train_path,
                                batch_size=FLAGS.batch_size,
                                img_size=FLAGS.img_size)

        inference(image, num_classes, keep_prob):


def main():
    test_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--num_classes',
        type=int,
        default=14,
        help='Number of classes for the dataset.'
    )
    parser.add_argument(
        '--img_size',
        type=int,
        default=64,
        help='Size of the image to be fed to the model.'
    )
    parser.add_argument(
        '--test_path',
        type=str,
        default='tfrecord/test.tfrecords',
        help='Path to the test data (tfrecords file).'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='ckpt_00',
        help='Directory for load the model checkpoints.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


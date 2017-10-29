from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import tensorflow as tf
import block_cnn as blocks

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
                                        num_threads=2)

    return images, labels


def test_model():
    with tf.Graph().as_default():
        images, labels = inputs(filename=FLAGS.test_path,
                                batch_size=FLAGS.batch_size,
                                img_size=FLAGS.img_size)

        logits = blocks.inference(image=images,
                                  num_classes=FLAGS.num_classes,
                                  keep_prob=FLAGS.keep_prob)

        loss = blocks.loss(logits, labels)

        train_op = blocks.train(loss, FLAGS.learning_rate)

        logits = blocks.inference(image=images,
                                  num_classes=FLAGS.num_classes,
                                  keep_prob=1.0)

        accuracy = blocks.evaluation(logits=logits,
                                     labels=labels)

        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Checkpoint saver (save the model)
        saver = tf.train.Saver()

        # Set checkpoint path and restore checkpoint if exists
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            print('Loaded model from latest checkpoint')

        # Coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,
                                               coord=coord)

        try:
            while not coord.should_stop():
                # Train network, compute loss and summaries for TensorBoard
                accuracy = sess.run(accuracy)
        except tf.errors.OutOfRangeError:
            # End of training
            print('Done with test')
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()



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
        default='ckpt',
        help='Directory for load the model checkpoints.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


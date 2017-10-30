from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import sys
import time
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

    # Data argumentation
    image = tf.image.random_flip_left_right(image)

    # Convert to between [-0.5, 0.5]
    image = image * (1.0 / 255) - 0.5

    # Convert label to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(filename, batch_size, num_epochs, img_size):
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
    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=num_epochs,
                                                    shuffle=True)

    image, label = read_and_decode(filename_queue, img_size)

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=batch_size,
                                            capacity=100 + 3*batch_size,
                                            num_threads=1,
                                            min_after_dequeue=100)
    return images, labels


def train_model():
    """
    Function to train the model
    """
    with tf.Graph().as_default():
        # Graph definition
        with tf.name_scope('input'):
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

            train_images, train_labels = inputs(filename=FLAGS.train_path,
                                                batch_size=FLAGS.batch_size,
                                                num_epochs=FLAGS.num_epochs,
                                                img_size=FLAGS.img_size)

            valid_images, valid_labels = inputs(filename=FLAGS.validation_path,
                                                batch_size=FLAGS.batch_size,
                                                num_epochs=FLAGS.num_epochs,
                                                img_size=FLAGS.img_size)

            _images = tf.placeholder_with_default(input=train_images,
                                                  shape=[None,
                                                         FLAGS.img_size,
                                                         FLAGS.img_size,
                                                         3],
                                                  name='images')

            _labels = tf.placeholder_with_default(input=train_labels,
                                                  shape=[None],
                                                  name='labels')

            keep_prob = tf.placeholder_with_default(FLAGS.keep_prob,
                                                    shape=[],
                                                    name='keep_prob')

        logits = blocks.inference(image=_images,
                                  num_classes=FLAGS.num_classes,
                                  keep_prob=keep_prob)

        loss = blocks.loss(logits, _labels)

        train_op = blocks.train(loss, FLAGS.learning_rate, global_step)

        # Summary for TensorBoard
        summary_op = tf.summary.merge_all()

        with tf.name_scope('evaluate'):
            accuracy = blocks.evaluation(logits, _labels)
            training_summary = tf.summary.scalar("training_accuracy", accuracy)
            validation_summary = tf.summary.scalar("validation_accuracy", accuracy)

        # Session
        sess = tf.Session()
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        sess.run(init_op)

        # Checkpoint saver (save the model)
        saver = tf.train.Saver(max_to_keep=5)

        # Set checkpoint path and restore checkpoint if exists
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, save_path=ckpt.model_checkpoint_path)
            print('Loaded model from latest checkpoint')

        # Writer for TensorBoard
        os.makedirs(FLAGS.tensorboard_dir, exist_ok=True)
        writer = tf.summary.FileWriter(FLAGS.tensorboard_dir, sess.graph)

        # Coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,
                                               coord=coord)

        try:
            step = sess.run(global_step)
            start_time = time.time()

            while not coord.should_stop():
                # Train network, compute loss and summaries for TensorBoard
                _, loss_value, summary = sess.run([train_op, loss, summary_op])
                writer.add_summary(summary, global_step=step)

                # Save checkpoint and print training update
                if step % 100 == 0:
                    # Save checkpoint
                    saver.save(sess, FLAGS.checkpoint_dir + "/block_cnn")

                    # Training accuracy
                    train_acc, train_summary = sess.run([accuracy, training_summary],
                                                        feed_dict={keep_prob: 1.0})
                    writer.add_summary(train_summary, step)

                    # Validation accuracy
                    vimage, vlabels = sess.run([valid_images, valid_labels])
                    valid_acc, valid_summary = sess.run([accuracy, validation_summary],
                                                        feed_dict={_images: vimage,
                                                                   _labels: vlabels,
                                                                   keep_prob: 1.0})
                    writer.add_summary(valid_summary, step)

                    duration = time.time() - start_time
                    print('Step %d | Loss = %.2f | Train Accuracy = %.2f | Validation Accuracy = %.2f (%.3f sec)'
                          % (step, loss_value, train_acc, valid_acc, duration))

                step += 1
        except tf.errors.OutOfRangeError:
            # End of training
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
            saver.save(sess, FLAGS.checkpoint_dir + "/block_cnn")
        finally:
            coord.request_stop()

        coord.join(threads)
        sess.close()


def main(_):
    train_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.01,
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--keep_prob',
        type=float,
        default=0.8,
        help='Keep probability for drop out.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1000,
        help='Number of epochs to run trainer.'
    )
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
        '--batch_size',
        type=int,
        default=128,
        help='Batch size.'
    )
    parser.add_argument(
        '--train_path',
        type=str,
        default='tfrecords/train.tfrecords',
        help='Directory with the training data.'
    )
    parser.add_argument(
        '--validation_path',
        type=str,
        default='tfrecords/validation.tfrecords',
        help='Directory with the validation data.'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='ckpt',
        help='Directory for save/load the model checkpoints.'
    )
    parser.add_argument(
        '--tensorboard_dir',
        type=str,
        default='tmp',
        help='Directory for saving the TensorBoard summaries.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

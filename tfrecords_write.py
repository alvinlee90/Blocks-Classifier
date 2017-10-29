from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import tensorflow as tf
import numpy as np

from random import shuffle

# Path for the dataset
dataset_path = 'dataset'

# Files for the tfrecords
train_filename = 'tfrecord/train.tfrecords'
val_filename = 'tfrecord/validation.tfrecords'
test_filename = 'tfrecord/test.tfrecords'

# Image extension
image_ext = '.jpeg'

# All classes for the dataset
classes = ['Blue Cube',
           'Blue Hollow Triangle',
           'Green Cube',
           'Green Hollow Cube',
           'Green Hollow Cylinder',
           'Orange Hollow Cross',
           'Orange Star',
           'Purple Hollow Cross',
           'Purple Star',
           'Red Hollow Cube',
           'Red Hollow Cylinder',
           'Red Sphere',
           'Yellow Cube',
           'Yellow Sphere']


def _int64_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a byte array
    '''
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    '''
    Creates a TensorFlow Record Feature with value as a 64 bit integer.
    '''
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_image(addrs):
    """
    Read an image, crop to square aspect ratio (480 x 480), resize to 100, 100

    Args:
        addrs: relative address of the image

    Returns:
        img: processed image
    """
    img = cv2.imread(addrs)
    img = img[0:480, 70:550]
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    return img


def write_to_tfrecord(dataset, addrs, labels, tfrecord_name):
    writer = tf.python_io.TFRecordWriter(tfrecord_name)

    for i in range(len(addrs)):
        # Load the image
        img = load_image(addrs[i])
        label = labels[i]

        # Create a feature
        feature = {'label': _int64_feature(label),
                   'image_raw': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    print('Completed writing TFRecord for ' + dataset + ' dataset')


def main():
    # Read addresses and create labels from the dataset
    addrs = [os.path.join(dirpath, f)
             for dirpath, dirnames, files in os.walk(dataset_path)
             for f in files if f.endswith(image_ext)]

    labels = []
    for addr in addrs:
        for index, shape in enumerate(classes):
            if shape in addr:
                labels.append(index)
                continue

    # Shuffle the images
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

    # Divide the data into 80% train, 10% validation, and 10% test
    train_addrs = addrs[0:int(0.8*len(addrs))]
    train_labels = labels[0:int(0.8*len(labels))]

    val_addrs = addrs[int(0.8*len(addrs)):int(0.9*len(addrs))]
    val_labels = labels[int(0.8*len(addrs)):int(0.9*len(addrs))]

    test_addrs = addrs[int(0.9*len(addrs)):]
    test_labels = labels[int(0.9*len(labels)):]

    write_to_tfrecord('train', train_addrs, train_labels, train_filename)
    write_to_tfrecord('val', val_addrs, val_labels, val_filename)
    write_to_tfrecord('test', test_addrs, test_labels, test_filename)


if __name__ == "__main__":
    main()

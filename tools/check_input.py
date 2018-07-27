import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
import os
import json
from bunch import Bunch
import matplotlib.pyplot as plt
import argparse
import numpy as np


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        metavar='C',
        default='None',
        help='The Configuration file')
    argparser.add_argument(
        '-i', '--input',
        metavar='C',
        default='None',
        help='The Input file')
    args = argparser.parse_args()
    return args


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)

    return config, config_dict


def process_config(json_file):
    config, _ = get_config_from_json(json_file)
    config.model_dir = os.path.join("../experiments", config.exp_name, "model/")
    config.tensorboard_dir = os.path.join("../experiments", config.exp_name, "log/")
    return config



def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)


def load_pair_paths(config):
    pair_paths = sorted(
        [os.path.join(config.data_folder, x) for x in os.listdir(config.data_folder) if x.endswith('.pkl')])
    return pair_paths


def calculate_num_iter(config, data):
    return int(len(data) / config.batch_size)


def split_dataset(config, dataset):
    return train_test_split(dataset,
                            test_size=config.test_split, random_state=42,
                            shuffle=True)


def tf_data_generator(config, pair_paths, is_training):
    def _read_file(pair_path):

        fp = open(pair_path, 'rb')
        data = pickle.load(fp)

        return data['image'], data['depth']

    def _set_shapes(image, depth):

        image.set_shape([640, 480, 3])
        depth.set_shape([640, 480, 1])

        return image, depth

    def _normalize_data(image, depthmap):
        image = tf.cast(image, tf.float32)
        image = image / 127.5
        image -= 1.

        depthmap = tf.cast(depthmap, tf.float32)
        depthmap = tf.div(
            tf.subtract(
                depthmap,
                tf.reduce_min(depthmap)
            ),
            tf.subtract(
                tf.reduce_max(depthmap),
                tf.reduce_min(depthmap)
            )
        )
        # depthmap = depthmap / 255.0
        # depthmap = tf.cast(depthmap, tf.int32)

        return image, depthmap

    def _flip_left_right(image, depthmap):
        seed = random.random()
        image = tf.image.random_flip_left_right(image, seed=seed)
        depth_map = tf.image.random_flip_left_right(depthmap, seed=seed)

        return image, depth_map

    def _resize_data(image, depthmap):
        image = tf.image.resize_images(image, config.input_size[:2])
        depthmap = tf.image.resize_images(depthmap, config.output_size[:2])
        return image, depthmap

    pair_tensor = tf.constant(pair_paths)

    # image_tensor = tf.placeholder(tf.float32, shape=[None] + config.input_size)
    # depth_tensor = tf.placeholder(tf.float32, shape=[None] + config.output_size)

    dataset = tf.data.Dataset.from_tensor_slices(pair_tensor)

    dataset = dataset.map(
        lambda pair_path: tuple(tf.py_func(
            _read_file, [pair_path], [tf.uint8, tf.float32])))

    # dataset = tf.data.Dataset.from_tensor_slices((images, depths))
    if is_training:
        dataset = dataset.shuffle(len(pair_paths))  # depends on sample size

    dataset = dataset.map(_set_shapes, num_parallel_calls=config.num_threads)
    dataset = dataset.map(_resize_data, num_parallel_calls=config.num_threads)
    if config.augment and is_training:
        dataset = dataset.map(_flip_left_right,
                              num_parallel_calls=config.num_threads)

    dataset = dataset.map(_normalize_data,
                          num_parallel_calls=config.num_threads)

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset


if __name__ == '__main__':

    args = get_args()
    config = process_config(args.config)
    dataset = load_pair_paths(config)
    train_pairs, test_pairs = split_dataset(config, dataset)

    train_gen = tf_data_generator(config, train_pairs, is_training=True)
    test_gen = tf_data_generator(config,test_pairs, is_training=False)

    iterator = train_gen.make_one_shot_iterator()

    image, depth = iterator.get_next()

    with tf.Session() as sess:
        # Evaluate the tensors
        aug_image, aug_depth = sess.run([image, depth])

        # Confirming everything is working by visualizing
        plt.figure('augmented image')
        plt.imshow(aug_image[0, :, :, :])
        plt.figure('augmented depth')
        aug_depth = aug_depth[0, :, :]
        plt.imshow(np.reshape(aug_depth, [aug_depth.shape[0], aug_depth.shape[1]]))
        plt.show()


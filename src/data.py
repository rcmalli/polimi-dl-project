import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
import os


def load_pair_paths(config):
    pair_paths = sorted([os.path.join(config.data_folder, x) for x in os.listdir(config.data_folder) if x.endswith('.pkl')])
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

    def _set_shapes(image,depth):

        image.set_shape([640, 480, 3])
        depth.set_shape([640, 480, 1])

        return image, depth


    def _normalize_data(image, depthmap):
        image = tf.cast(image, tf.float32)
        image = image / 127.5
        image -= 1.

        depthmap = tf.cast(depthmap, tf.float32)
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

    #dataset = tf.data.Dataset.from_tensor_slices((images, depths))
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

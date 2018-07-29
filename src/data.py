import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split
import os






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

        if config.normalize_depth:
            depthmap = depthmap / 10.0

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

    def _corrupt_brightness(image, depth):
        cond_brightness = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_hue(
            image, 0.1), lambda: tf.identity(image))
        return image, depth

    def _corrupt_contrast(image, depth):
        cond_contrast = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, depth

    def _corrupt_saturation(image, depth):
        cond_saturation = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(
            image, 0.2, 1.8), lambda: tf.identity(image))
        return image, depth

    def _crop_random(image, depth):

        seed = random.random()
        cond_crop_image = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
        cond_crop_depth = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

        image = tf.cond(cond_crop_image, lambda: tf.random_crop(
            image, [int(640 * 0.9), int(480 * 0.9), 3], seed=seed), lambda: tf.identity(image))
        depth = tf.cond(cond_crop_depth, lambda: tf.random_crop(
            depth, [int(640 * 0.9), int(480 * 0.9), 1], seed=seed), lambda: tf.identity(depth))
        image = tf.expand_dims(image, axis=0)
        depth = tf.expand_dims(depth, axis=0)

        image = tf.image.resize_images(image, [640, 480])
        depth = tf.image.resize_images(depth, [640, 480])

        image = tf.squeeze(image, axis=0)
        depth = tf.squeeze(depth, axis=0)

        return image, depth

    def _rotate_random(image, depth):
        seed = random.random()
        cond_rotate_image = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)
        cond_rotate_depth = tf.cast(tf.random_uniform(
            [], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

        image = tf.cond(cond_rotate_image, lambda: tf.contrib.image.rotate(image, 0.17453),
                        lambda: tf.identity(image))  # ~10 degree
        depth = tf.cond(cond_rotate_depth, lambda: tf.contrib.image.rotate(depth, 0.17453),
                        lambda: tf.identity(depth))  # ~10 degree

        return image, depth

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

    dataset = dataset.repeat()
    dataset = dataset.map(_set_shapes, num_parallel_calls=config.num_threads)

    if is_training:
        if config.flip_left_right:
            dataset = dataset.map(_flip_left_right,
                                  num_parallel_calls=config.num_threads)
        if config.corrupt_saturation:
            dataset = dataset.map(_corrupt_saturation,
                                  num_parallel_calls=config.num_threads)
        if config.corrupt_brightness:
            dataset = dataset.map(_corrupt_brightness,
                                  num_parallel_calls=config.num_threads)
        if config.corrupt_contrast:
            dataset = dataset.map(_corrupt_contrast,
                                  num_parallel_calls=config.num_threads)
        if config.random_crop:
            dataset = dataset.map(_crop_random,
                                  num_parallel_calls=config.num_threads)
        if config.random_rotate:
            dataset = dataset.map(_rotate_random,
                                  num_parallel_calls=config.num_threads)


    dataset = dataset.map(_resize_data, num_parallel_calls=config.num_threads)
    dataset = dataset.map(_normalize_data,
                          num_parallel_calls=config.num_threads)

    dataset = dataset.batch(config.batch_size)

    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

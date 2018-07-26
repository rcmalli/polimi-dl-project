import tensorflow as tf
import random
import pickle
from sklearn.model_selection import train_test_split

def load_dataset_file(config):
    with open(config.data_folder + '/nyudataset.pickle', 'rb') as handle:
        data = pickle.load(handle)
        return data


def calculate_num_iter(config, data):

    return int(data.shape[0] / config.batch_size)


def split_dataset(config, dataset):
    return train_test_split(dataset['image'], dataset['depth'],
                            test_size=config.test_split, random_state=42,
                            shuffle=True)


def tf_data_generator(config, images, depths, is_training):
    def _normalize_data(image, depthmap):
        image = tf.cast(image, tf.float32)
        image = image / 255.0

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

    # image_tensor = tf.placeholder(tf.float32, shape=[None] + list(images.shape[1:]))
    # depth_tensor = tf.placeholder(tf.float32, shape=[None] + list(depths.shape[1:]))

    # image_tensor = tf.placeholder(tf.float32, shape=[None] + config.input_size)
    # depth_tensor = tf.placeholder(tf.float32, shape=[None] + config.output_size)

    # dataset = tf.data.Dataset.from_tensor_slices((image_tensor, depth_tensor))

    dataset = tf.data.Dataset.from_tensor_slices((images, depths))
    if is_training:
        dataset = dataset.shuffle(images.shape[0])  # depends on sample size

    dataset = dataset.map(_resize_data, num_parallel_calls=config.num_threads).prefetch(30)
    if config.augment and is_training:
        dataset = dataset.map(_flip_left_right,
                              num_parallel_calls=config.num_threads).prefetch(30)

    dataset = dataset.map(_normalize_data,
                          num_parallel_calls=config.num_threads).prefetch(30)

    dataset = dataset.batch(config.batch_size)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)

    return dataset

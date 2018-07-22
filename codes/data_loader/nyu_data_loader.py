import numpy as np
import h5py
import tensorflow as tf
import random
import os
import pickle


def _resize_data(image, depthmap):
    """Resizes images to smaller dimensions."""
    image = tf.image.resize_images(image, [48, 64])
    depthmap = tf.image.resize_images(depthmap, [24, 32])

    return image, depthmap

def _flip_left_right(image, depthmap):
    """Randomly flips image and depth_map left or right in accord."""
    seed = random.random()
    image = tf.image.random_flip_left_right(image, seed=seed)
    depth_map = tf.image.random_flip_left_right(depthmap, seed=seed)

    return image, depth_map


def _normalize_data(image, depthmap):
    """Normalize image and depth_map within range 0-1."""
    image = tf.cast(image, tf.float32)
    #image = image / 255.0

    depthmap = tf.cast(depthmap, tf.float32)
    #depthmap = depthmap / 255.0
    #depthmap = tf.cast(depthmap, tf.int32)

    return image, depthmap


def _parse_depthmap(image_path, depthmap_path):

    fp = open(depthmap_path, 'rb')
    depthmap = pickle.load(fp)
    # depthmap = np.load(depthmap_path)
    depthmap = depthmap.astype(np.float32)
    depthmap = np.expand_dims(depthmap, axis=-1)
    # fp.close()

    return image_path, depthmap


def _parse_image(image_path, depthmap):

    """Reads image and depth_map files"""
    image_content = tf.read_file(image_path)
    image = tf.image.decode_png(image_content, channels=3)

    return image, depthmap


class NYUDataLoader:
    def __init__(self, config):
        self.config = config
        print("Loading the data")
        image_folder = self.config.image_folder
        depthmap_folder = self.config.depthmap_folder
        self.augment = self.config.augment
        self.batch_size = self.config.batch_size
        self.num_threads = self.config.num_threads

        self.image_paths = sorted([os.path.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.jpeg')])
        self.depthmap_paths = sorted([os.path.join(depthmap_folder, x) for x in os.listdir(depthmap_folder) if x.endswith('.pkl')])

        images_name_tensor = tf.constant(self.image_paths)
        mask_name_tensor = tf.constant(self.depthmap_paths)

        # Create dataset out of the 2 files:
        self.data = tf.data.Dataset.from_tensor_slices(
            (images_name_tensor, mask_name_tensor))

        self.data = self.data.map(
            lambda image_path, depthmap_path: tuple(tf.py_func(
                _parse_depthmap, [image_path, depthmap_path], [image_path.dtype, tf.float32])))

        # Parse images
        self.data = self.data.map(
            _parse_image, num_parallel_calls=self.num_threads).prefetch(30)

        # If augmentation is to be applied
        if self.augment:
            self.data = self.data.map(_flip_left_right,
                            num_parallel_calls=self.num_threads).prefetch(30)

        # Batch the data
        self.data = self.data.batch(self.batch_size)

        # Resize to smaller dims for speed
        self.data = self.data.map(_resize_data, num_parallel_calls=self.num_threads).prefetch(30)

        # Normalize
        self.data = self.data.map(_normalize_data,
                        num_parallel_calls=self.num_threads).prefetch(30)

        self.data = self.data.shuffle(30)

        # Create iterator
        self.iterator = tf.data.Iterator.from_structure(
            self.data.output_types, self.data.output_shapes)

        # Data set init. op
        self.iter_init_op = self.iterator.make_initializer(self.data)
        self.x, self.y = self.iterator.get_next()


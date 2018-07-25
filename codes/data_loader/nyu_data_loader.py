import numpy as np
import h5py
import tensorflow as tf
import random
import os
import pickle
from sklearn.model_selection import train_test_split





class NYUDataLoader:
    def __init__(self, config):
        self.config = config
        print("Loading the data")
        data_folder = self.config.data_folder
        self.augment = self.config.augment
        self.batch_size = self.config.batch_size
        self.num_threads = self.config.num_threads

        with open(data_folder+'/nyudataset.pickle', 'rb') as handle:
            data = pickle.load(handle)

        print("image array: ",data['image'].shape)
        print("depth array: ", data['depth'].shape)

        self.train_images, self.test_images, self.train_depths, self.test_depths = train_test_split(
            data['image'], data['depth'], test_size=self.config.test_split, random_state=42, shuffle=True)

        self.config.test_num_iter_per_epoch = int(self.test_images.shape[0]/ self.config.batch_size)
        self.config.train_num_iter_per_epoch = int(self.train_images.shape[0]/ self.config.batch_size)

        self.image = tf.placeholder(tf.float32, shape=[None] + list(self.train_images.shape[1:]))
        self.depth = tf.placeholder(tf.float32, shape=[None] + list(self.train_depths.shape[1:]))

        self.dataset = tf.data.Dataset.from_tensor_slices((self.image, self.depth))

        self.dataset = self.dataset.map(self._resize_data, num_parallel_calls=self.num_threads).prefetch(30)
        if self.augment:
            self.dataset = self.dataset.map(self._flip_left_right,
                                                  num_parallel_calls=self.num_threads).prefetch(30)
        self.dataset = self.dataset.map(self._normalize_data,
                                              num_parallel_calls=self.num_threads).prefetch(30)
        self.dataset = self.dataset.shuffle(30)
        self.dataset =self.dataset.batch(self.batch_size)

        # Create iterator
        self.iterator = self.dataset.make_initializable_iterator()
        self.x, self.y = self.iterator.get_next()



    def _resize_data(self, image, depthmap):
        """Resizes images to smaller dimensions."""
        image = tf.image.resize_images(image, self.config.input_size[:2])
        depthmap = tf.image.resize_images(depthmap, self.config.output_size[:2])

        return image, depthmap

    def _flip_left_right(self, image, depthmap):
        """Randomly flips image and depth_map left or right in accord."""
        seed = random.random()
        image = tf.image.random_flip_left_right(image, seed=seed)
        depth_map = tf.image.random_flip_left_right(depthmap, seed=seed)

        return image, depth_map

    def _normalize_data(self, image, depthmap):
        """Normalize image and depth_map within range 0-1."""
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        depthmap = tf.cast(depthmap, tf.float32)
        # depthmap = depthmap / 255.0
        # depthmap = tf.cast(depthmap, tf.int32)

        return image, depthmap



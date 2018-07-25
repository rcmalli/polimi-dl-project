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

        train_images, test_images, train_depths, test_depths = train_test_split(
            data['image'], data['depth'], test_size=self.config.test_split, random_state=42, shuffle=True)

        self.config.test_num_iter_per_epoch = int(test_images.shape[0]/ self.config.batch_size)
        self.config.train_num_iter_per_epoch = int(train_images.shape[0]/ self.config.batch_size)




        with tf.name_scope('train_dataset'):
            self.train_data = tf.data.Dataset.from_tensor_slices(
                (train_images, train_depths))



            self.train_data = self.train_data.map(self._resize_data, num_parallel_calls=self.num_threads).prefetch(30)
            if self.augment:
                self.train_data = self.train_data.map(self._flip_left_right,
                                num_parallel_calls=self.num_threads).prefetch(30)
            self.train_data = self.train_data.map(self._normalize_data,
                            num_parallel_calls=self.num_threads).prefetch(30)
            self.train_data = self.train_data.shuffle(30)
            self.train_data = self.train_data.batch(self.batch_size)

        with tf.name_scope('test_dataset'):
            self.test_data = tf.data.Dataset.from_tensor_slices(
                (test_images, test_depths))

            self.test_data = self.test_data.map(self._resize_data, num_parallel_calls=self.num_threads).prefetch(30)
            self.test_data = self.test_data.map(self._normalize_data,
                            num_parallel_calls=self.num_threads).prefetch(30)
            self.test_data = self.test_data.shuffle(30)
            self.test_data = self.test_data.batch(self.batch_size)

        # Create iterator
        self.iterator = tf.data.Iterator.from_structure(
            self.train_data.output_types, self.train_data.output_shapes)

        # Data set init. op
        self.train_init_op = self.iterator.make_initializer(self.train_data)
        self.test_init_op = self.iterator.make_initializer(self.test_data)
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



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
from src.config import process_config
from utils import get_args
from src.data import load_pair_paths,split_dataset,tf_data_generator
from src.model import depth_model_v4


if __name__ == '__main__':


    args = get_args()
    config = process_config(args.config)
    dataset = load_pair_paths(config)

    model = depth_model_v4(config)
    config.output_size = list(model.output_shape[1:])

    train_pairs, test_pairs = split_dataset(config, dataset)

    train_gen = tf_data_generator(config, train_pairs, is_training=True)

    iterator = train_gen.make_one_shot_iterator()

    image, depth = iterator.get_next()

    with tf.Session() as sess:
        # Evaluate the tensors
        aug_image, aug_depth = sess.run([image, depth])

        # Confirming everything is working by visualizing
        plt.figure('augmented image')
        plt.imshow(((aug_image[0, :, :, :]+1) *127.5).astype(np.uint8))
        plt.figure('augmented depth')
        aug_depth = aug_depth[0, :, :]
        plt.imshow(np.reshape(aug_depth, [aug_depth.shape[0], aug_depth.shape[1]]))
        plt.show()


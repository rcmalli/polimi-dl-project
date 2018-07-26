import argparse
import numpy as np
import tensorflow as tf
import os
from PIL import Image
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from models.resnet import Resnet50Model
from models.mini_model import MiniModel
import matplotlib.pyplot as plt


# import matplotlib.pyplot as plt
def predict(input_path, config):
    img = Image.open(input_path)

    height = config.input_size[1]
    width = config.input_size[2]
    channels = config.input_size[3]

    img = img.resize((width, height), Image.ANTIALIAS)
    img = np.array(img, dtype=float)

    #	plt.imshow(np.asarray(img))
    #	plt.show()

    img = np.reshape(img, [1, width, height, channels])
    img = (img) / 255
    print(img.shape)
    x = tf.placeholder(tf.float32, shape=config.input_size)
    y = tf.placeholder(tf.float32, shape=config.output_size)

    #	x = tf.constant(width)
    #	y = tf.constant(height)
    with tf.Session() as sess:
        sess.as_default()
        model = Resnet50Model(config, sess, x, y)
        tf.global_variables_initializer().run()
        tf.tables_initializer().run()
        model.load(sess)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            pred = sess.run(model.output, feed_dict={model.x: img, model.is_training: False})
            pred = np.reshape(pred, [pred.shape[1], pred.shape[2]])

    return pred


def main():
    args = get_args()
    config = process_config(args.config)

    pred = predict(args.input, config)
    plt.imsave('lena_depth.jpeg', pred)
    #	plt.imshow(np.asarray(pred),cmap='gray')
    #	plt.show()
    os._exit(0)


if __name__ == '__main__':
    main()

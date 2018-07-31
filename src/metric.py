from tensorflow.keras import backend as K
import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def abs_relative(target, pred):
    # Get absolute error for each pixel in batch
    rel_error = tf.abs(tf.subtract(target, pred), name='abs_error') / target

    return tf.reduce_mean(rel_error)


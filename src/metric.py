from tensorflow.keras import backend as K
import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def abs_relative(target, pred):
    # Get absolute error for each pixel in batch
    rel_error = tf.abs(tf.subtract(target, pred), name='abs_error') / target

    return tf.reduce_sum(rel_error)

def t_relative(target, pred):
    # Get relative error for each pixel in batch
    t = tf.maximum(tf.divide(target, pred),tf.divide(pred,target), name='t_relative') 
    # Find pixels relative error is less than 1.25
    t = tf.less(t, tf.constant(1.25))
    # Cast to int
    t = tf.cast(t, tf.int32)
    # Count pixels
    count = tf.reduce_sum(t)
    return tf.divide(count,tf.size(t))

import tensorflow as tf
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

def simse_create(config):
    def simse(target, pred):
        lamda = tf.constant(config.lamda)
        return tf.subtract(
            tf.reduce_mean(tf.square(tf.subtract(tf.log1p(target), tf.log1p(pred)))),
            tf.multiply(lamda,
                        tf.square(tf.reduce_mean(tf.subtract(tf.log1p(target), tf.log1p(pred))))))

    return simse

def berhu(target, pred):
    
    # Get absolute error for each pixel in batch 
    abs_error = tf.abs(tf.subtract(target, pred), name='abs_error')

    # Calculate threshold c from max error
    c = 0.2 * tf.reduce_max(abs_error)
    # if, then, else
    berHu_loss = tf.where(abs_error <= c,   
                   abs_error, 
                  (tf.square(abs_error) + tf.square(c))/(2*c))
            
    return tf.reduce_mean(berHu_loss)







def huber(target, pred):
    return tf.reduce_mean(tf.losses.huber_loss(labels=target, predictions=pred))


def dummy_mse(target, pred):
    return tf.reduce_mean(tf.losses.mean_squared_error(labels=target, predictions=pred))


def select_loss(config):
    if config.loss_type == "HUBER":
        return huber
    elif config.loss_type == "BERHU":
        return berhu
    elif config.loss_type == "SIMSE":
        return simse_create(config)
    elif config.loss_type == "MAE":
        return mean_absolute_error
    else:
        return mean_squared_error

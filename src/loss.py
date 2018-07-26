import tensorflow as tf


def simse_create(config):
    def simse(target, pred):
        lamda = tf.constant(config.lamda)
        return tf.subtract(
            tf.reduce_mean(tf.square(tf.subtract(tf.log1p(target), tf.log1p(pred)))),
            tf.multiply(lamda,
                        tf.square(tf.reduce_mean(tf.subtract(tf.log1p(target), tf.log1p(pred))))))

    return simse


def huber(target, pred):
    return tf.reduce_mean(tf.losses.huber_loss(labels=target, predictions=pred))


def dummy_mse(target, pred):
    return tf.reduce_mean(tf.losses.mean_squared_error(labels=target, predictions=pred))


def select_loss(config):
    if config.loss_type == "HUBER":
        return huber
    elif config.loss_type == "SIMSE":
        return simse_create(config)
    else:
        return dummy_mse

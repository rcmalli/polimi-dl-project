from base.base_model import BaseModel
import tensorflow as tf


class MiniModel(BaseModel):
    def __init__(self, config, x, y):
        super(MiniModel, self).__init__(config)
        self.x = x
        self.y = y
        self.build_model()
        self.init_saver()


    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        convOut1 = tf.layers.conv2d(inputs=self.x, filters=8, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        # poolOut1 = tf.layers.max_pooling2d(inputs=convOut1, pool_size=[2, 2], strides=2)
        convOut2 = tf.layers.conv2d(inputs=convOut1, filters=1, kernel_size=[3, 3], padding="same",
                                    activation=tf.nn.relu)
        poolOut2 = tf.layers.max_pooling2d(inputs=convOut2, pool_size=[2, 2], strides=2)

        # Loss function
        with tf.name_scope("loss"):
            self.mse = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y, predictions=poolOut2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.mse,
                                                                                         global_step=self.global_step_tensor)
            ## TODO Need to be changed
            correct_prediction = tf.equal(tf.argmax(poolOut2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # self.accuracy = 0
        # Change the value of this parameter in case of changing architecture
        self.output = poolOut2

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

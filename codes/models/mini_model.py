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
        poolOut2 = tf.layers.max_pooling2d(inputs=convOut2, pool_size=[2, 2], strides=2, )

        # Put the last layer for the rest of the code
        outputLayer = poolOut2
        # Loss function
        with tf.name_scope("loss"):
            if (self.config.loss_type == "MSE"):
                self.loss_function = tf.reduce_mean(
                    tf.losses.mean_squared_error(labels=self.y, predictions=outputLayer))
            elif (self.config.loss_type == "HUBER"):
                self.loss_function = tf.reduce_mean(tf.losses.huber_loss(labels=self.y, predictions=outputLayer))
            elif (self.config.loss_type == "SIMSE"):
                # Scale invariant Mean Square Error
                self.lamda = tf.constant(self.config.lamda)
                # self.lamda = tf.multiply(self.lamda,tf.cast(tf.size(outputLayer),tf.float32))
                self.loss_function = tf.subtract(
                    tf.reduce_mean(tf.square(tf.subtract(tf.log1p(self.y), tf.log1p(outputLayer)))),
                    tf.multiply(self.lamda,
                                tf.square(tf.reduce_mean(tf.subtract(tf.log1p(self.y), tf.log1p(outputLayer))))))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss_function,
                                                                                         global_step=self.global_step_tensor)
            ## TODO Need to be changed
            correct_prediction = tf.equal(tf.argmax(outputLayer, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # self.accuracy = 0
        # Change the value of this parameter in case of changing architecture
        self.output = outputLayer

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

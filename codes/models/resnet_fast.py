from base.base_model import BaseModel
import tensorflow as tf
import tensorflow_hub as hub


class Resnet50Model(BaseModel):
    def __init__(self, config, session, x, y):
        super(Resnet50Model, self).__init__(config)
        self.session = session
        self.x = x
        self.y = y
        self.init_saver()
        self.build_model()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        # Full model scope
        with tf.variable_scope('DepthModel'):

            # resnet path
            with tf.variable_scope('resnet'):
                self.resnet50 = tf.keras.applications.ResNet50(weights='imagenet',
                                                               include_top=False, input_tensor=self.x)

                self.resnet_output = self.resnet50.get_layer('activation_48').output

            # up projection path
            with tf.variable_scope('up_projection'):
                # add more layers starting from self.resnet_output
                print("TODO: add other layers")

                # Dummy layers for testing
                # Deconvolution, image shape: (batch, 14, 14, 64)
                x = tf.layers.conv2d_transpose(self.resnet_output, 64, 4, strides=2)
                # Deconvolution, image shape: (batch, 28, 28, 1)
                x = tf.layers.conv2d_transpose(x, 64, 4, strides=2)

                self.output_layer = x


        # variables that should be excluded in training
        self.resnet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DepthModel/resnet')

        self.model_weights_tensors = set(self.resnet_weights)

        # Loss function
        with tf.name_scope("loss"):
            if self.config.loss_type == "MSE":
                self.loss_function = tf.reduce_mean(
                    tf.losses.mean_squared_error(labels=self.y, predictions=self.output_layer))
            elif self.config.loss_type == "HUBER":
                self.loss_function = tf.reduce_mean(tf.losses.huber_loss(labels=self.y, predictions=self.output_layer))
            elif self.config.loss_type == "SIMSE":
                # Scale invariant Mean Square Error
                self.lamda = tf.constant(self.config.lamda)
                # self.lamda = tf.multiply(self.lamda,tf.cast(tf.size(outputLayer),tf.float32))
                self.loss_function = tf.subtract(
                    tf.reduce_mean(tf.square(tf.subtract(tf.log1p(self.y), tf.log1p(self.output_layer)))),
                    tf.multiply(self.lamda,
                                tf.square(tf.reduce_mean(tf.subtract(tf.log1p(self.y), tf.log1p(self.output_layer))))))
            non_resnet_vars = list(
                set(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) - self.model_weights_tensors)
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss_function,
                                                                                         var_list=non_resnet_vars,
                                                                                         global_step=self.global_step_tensor)
            ## TODO Need to be changed
            correct_prediction = tf.equal(tf.argmax(self.output_layer, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # self.accuracy = 0
        # Change the value of this parameter in case of changing architecture
        self.output = self.output_layer

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

from base.base_model import BaseModel
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    elif type(incoming) in [np.array, list, tuple]:
        return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)


def unpool_as_conv(size, input_data, id, stride=1, ReLU=False, BN=True, is_training=False):
    with tf.variable_scope('unpool_' + id):
        # Model upconvolutions (unpooling + convolution) as interleaving feature
        # maps of four convolutions (A,B,C,D). Building block for up-projections.

        layerName = "layer%s_ConvA" % (id)
        outputA = tf.layers.conv2d(input_data, size[3], 3, strides=(stride, stride),
                                       activation=None, name=layerName, padding='same')
        layerName = "layer%s_ConvB" % (id)
        padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
        outputB = tf.layers.conv2d(padded_input_B, size[3], (2, 3), strides=(stride, stride),
                                       activation=None, name=layerName, padding='valid')
        layerName = "layer%s_ConvC" % (id)
        padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
        outputC = tf.layers.conv2d(padded_input_C, size[3], (3, 2), strides=(stride, stride),
                                       activation=None, name=layerName, padding='valid')
        layerName = "layer%s_ConvD" % (id)
        padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
        outputD = tf.layers.conv2d(padded_input_D, size[3], (2, 2), strides=(stride, stride),
                                       activation=None, name=layerName, padding='valid')

        # interleave_name = "interleave_%s" % (id)
        # with tf.name_scope(interleave_name):
        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = interleave([outputA, outputB], axis=1)  # columns
        right = interleave([outputC, outputD], axis=1)  # columns
        Y = interleave([left, right], axis=2)  # rows

        if BN:
            # print("Check batchnorm implementation")
            layerName = "layer%s_BN" % (id)
            Y = tf.layers.batch_normalization(Y, training=is_training, name=layerName)

        if ReLU:
            Y = tf.nn.relu(Y, name=layerName)

        return Y


def up_project(input_data, size, id, stride=1, BN=True, is_training=False):
    # Create residual upsampling layer (UpProjection)

    with tf.variable_scope('up_project_' + id):
        # Branch 1
        id_br1 = "%s_br1" % (id)

        # Interleaving Convs of 1st branch
        branch1_output = unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, BN=BN, is_training=is_training)

        layerName = "layer%s_Conv" % (id)
        # Convolution following the upProjection on the 1st branch
        branch1_output = tf.layers.conv2d(branch1_output, size[3], (size[0], size[1]), strides=(stride, stride),
                                          activation=None, name=layerName, padding='same')

        if BN:
            layerName = "layer%s_BN" % (id)
            branch1_output = tf.layers.batch_normalization(branch1_output, training=is_training, name=layerName)

        # Branch 2
        id_br2 = "%s_br2" % (id)
        # Interleaving convolutions and output of 2nd branch
        branch2_output = unpool_as_conv(size, input_data, id_br2, stride, ReLU=False, BN=BN, is_training=True)

        # sum branches
        layerName = "layer%s_Sum" % (id)
        output = tf.add_n([branch1_output, branch2_output], name=layerName)
        # ReLU
        layerName = "layer%s_ReLU" % (id)
        output = tf.nn.relu(output, name=layerName)

        return output


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
                # print("TODO: check upscale part")
                x = tf.layers.conv2d(self.resnet_output, 1024, 1, strides=(1, 1), activation=tf.nn.relu, name='layer1',
                                     padding='same')
                x = tf.layers.batch_normalization(x, training=self.is_training, name='layer1_BN')
                x = up_project(x, [3, 3, 1024, 512], id='2x', stride=1, BN=True, is_training=self.is_training)
                x = up_project(x, [3, 3, 512, 256], id = '4x', stride=1, BN=True, is_training=self.is_training)
                x = up_project(x, [3, 3, 256, 128], id = '8x', stride=1, BN=True, is_training=self.is_training)
                x = up_project(x, [3, 3, 128, 64], id = '16x', stride=1, BN=True, is_training=self.is_training)
                x = tf.layers.dropout(x, rate=0.1, name='drop', training=self.is_training)
                x = tf.layers.conv2d(x, 1, 3, strides=(1, 1), activation=tf.nn.relu, name='ConvPred', padding='same')
                # What should be the output activation current is Relu
                self.output_layer = x

        # Uncomment this for debugging the graph
        # train_writer = tf.summary.FileWriter(self.config.checkpoint_dir)
        # train_writer.add_graph(self.session.graph)

        # put breakpoint at below line and open tensorboard to see graph.

        # variables that should be excluded in training
        self.resnet_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='DepthModel/resnet')

        self.model_weights_tensors = set(self.resnet_weights)

        # Loss function
        with tf.name_scope("loss_"+self.config.loss_type):
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

            # this list should include only upscale part
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

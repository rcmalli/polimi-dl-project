import numpy as np
import tensorflow as tf
from keras import backend as K
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, UpSampling2D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from loss import huber, simse_create, berhu


def load_depth_model(config):
    if config.loss_type == "HUBER":
        custom_object_dict = {'huber': huber}
    elif config.loss_type == "SIMSE":
        custom_object_dict = {'simse': simse_create}
    elif config.loss_type == "BERHU":
        custom_object_dict = {'berhu': berhu}
    else:
        custom_object_dict = {}

    model = load_model(config.model_dir + config.prediction_model_name, custom_objects=custom_object_dict)

    return model

def load_depth_model_from_weights(config):

    model = depth_model(config)
    model.load_weights(config.model_dir + config.prediction_model_name)
    return model





def unpool_resize(input):
    def unpool_resize_func(input_tensor):
        output = tf.image.resize_images(input_tensor, size=[int(input_tensor.shape[1] * 2), int(input_tensor.shape[1] * 2)],
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return output

    def unpool_output_shape(input_shape):
        return tuple([input_shape[0],int(input_shape[1] * 2),int(input_shape[2] * 2), int(input_shape[3])])

    return Lambda(unpool_resize_func, output_shape=unpool_output_shape)(input)



def unpool_simple(input):

    return UpSampling2D((2, 2))(input)

def unpool_deconv(input, size):

    return Conv2DTranspose(size, 2, (2, 2), padding='same')(input)

def unpool_checkerboard(input):

    def unpool_checkerboard_func(input_tensor):
        mask = np.ones((input_tensor.shape[1].value, input_tensor.shape[2].value, input_tensor.shape[3].value))
        mask[::2, :, :] = 0
        mask[:, ::2, :] = 0
        mask = 1 - mask
        mask_tensor = tf.constant(mask)
        mask_tensor = tf.cast(mask_tensor, tf.float32)
        output_tensor = tf.multiply(mask_tensor, input_tensor)
        return output_tensor

    def unpool_checkerboard_shape(input_shape):
        return input_shape

    return Lambda(unpool_checkerboard_func,output_shape=unpool_checkerboard_shape)(input)




def get_unpool(config, input, size):
    if config.unpool_type == "deconv":
        return unpool_deconv(input, size)
    elif config.unpool_type == "resize":

        return unpool_resize(input)
    elif config.unpool_type == "checkerboard":
        x = unpool_simple(input)
        return unpool_checkerboard(x)
    elif config.unpool_type == "simple":
        return unpool_simple(input)
    else:
       return NotImplementedError


def up_convolution(config, input, size):

    out = get_unpool(config, input, size)
    out = Conv2D(size, (5, 5), activation='relu', padding='same')(out)
    return out


def up_projection(config, input, size):

    up = get_unpool(config, input, size)
    x1 = Conv2D(size, (5, 5), activation='relu', padding='same')(up)
    if config.bn:
        x1 = BatchNormalization()(x1)
    x1 = Conv2D(size, (3, 3), activation=None, padding='same')(x1)
    if config.bn:
        x1 = BatchNormalization()(x1)

    x2 = Conv2D(size, (5, 5), activation=None, padding='same')(up)
    if config.bn:
        x2 = BatchNormalization()(x2)

    out = Add()([x1, x2])
    out = Activation('relu')(out)

    return out


def depth_model(config):


    def resnet(input_tensor):

        with K.name_scope('resnet50'):

            resnet_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                          include_top=False, input_tensor=input_tensor)

            if config.train_resnet:
                for layer in resnet_model.layers[:163]:
                    layer.trainable = False
            else:
                for layer in resnet_model.layers:
                    layer.trainable = False

            return resnet_model.get_layer('activation_48').output

    with K.name_scope('depthmodel'):

        input_tensor = Input(shape=config.input_size)
        resnet_out = resnet(input_tensor)

        with K.name_scope('upscale'):

            x = Conv2D(1024, (1, 1), activation="relu", name='layer1', padding='same')(resnet_out)
            x = BatchNormalization(name='layer1_bn')(x)
            for i in range(config.upscale):
                if config.model_type == "up_projection":
                    x = up_projection(config, x, int((2**(3-i))*64))
                elif config.model_type == "up_convolution":
                    x = up_convolution(config, x, int((2 ** (3 - i)) * 64))
                else :
                    return NotImplementedError
            out = Conv2D(1, 3, activation='relu', padding='same', name= 'conv_output')(x)

        model = Model(inputs=input_tensor, outputs=out)

    return model
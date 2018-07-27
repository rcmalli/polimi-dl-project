from unicodedata import name

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout, \
    ZeroPadding2D, Concatenate, Activation, Add, UpSampling2D
from tensorflow.keras.layers import Conv2DTranspose as DeConv
from tensorflow.keras.models import Model
from loss import dummy_mse, huber, simse_create
from tensorflow.keras.models import load_model
from keras import backend as K


def unpool_as_conv(size, input, id, stride=1, relu=False, bn=True):
    with K.name_scope('unpool_' + id):

        layerName = "layer%s_ConvA" % (id)

        outA = Conv2D(size[3], (3, 3), activation='relu', name=layerName, padding='same')(input)

        layerName = "layer%s_ConvB" % (id)
        outB = ZeroPadding2D(padding=((1, 0), (1, 1)))(input)
        outB = Conv2D(size[3], (2, 3), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(
            outB)

        layerName = "layer%s_ConvC" % (id)

        outC = ZeroPadding2D(padding=((1, 1), (1, 0)))(input)
        outC = Conv2D(size[3], (3, 2), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(
            outC)

        layerName = "layer%s_ConvD" % (id)

        outD = ZeroPadding2D(padding=((1, 0), (1, 0)))(input)
        outD = Conv2D(size[3], (2, 2), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(
            outD)

        left = Concatenate(axis=1)([outA, outB])
        right = Concatenate(axis=1)([outC, outD])
        out = Concatenate(axis=2)([left, right])

        if bn:
            layerName = "layer%s_BN" % (id)
            out = BatchNormalization(name=layerName)(out)

        if relu:
            out = Activation('relu')(out)

        return out


def up_project(input, size, id, stride=1, bn=True):
    with K.name_scope('up_project_' + id):
        id_br1 = "%s_br1" % (id)
        branch1_output = unpool_as_conv(size, input, id_br1, stride, relu=True, bn=bn)
        layerName = "layer%s_Conv" % (id)
        branch1_output = Conv2D(size[3], (size[0], size[1]), activation='relu', name=layerName,
                                strides=(stride, stride), padding='same')(branch1_output)

        if bn:
            layerName = "layer%s_BN_up" % (id)
            branch1_output = BatchNormalization(name=layerName)(branch1_output)

        id_br2 = "%s_br2" % (id)
        branch2_output = unpool_as_conv(size, input, id_br2, stride, relu=True, bn=bn)

        layerName = "layer%s_Sum" % (id)
        out = Add(name=layerName)([branch1_output, branch2_output])

        layerName = "layer%s_ReLU" % (id)
        out = Activation('relu', name=layerName)(out)

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

    input_tensor = Input(shape=config.input_size)
    x = resnet(input_tensor)

    with K.name_scope('upscaling'):

        x = Conv2D(1024, (1, 1), activation='relu', name='layer1', strides=(1, 1), padding='same')(x)
        x = BatchNormalization(name='layer1_bn')(x)
        x = up_project(x, [3, 3, 1024, 512], id='2x', stride=1, bn=True)
        x = up_project(x, [3, 3, 512, 256], id='4x', stride=1, bn=True)
        x = up_project(x, [3, 3, 256, 128], id='8x', stride=1, bn=True)
        x = up_project(x, [3, 3, 128, 64], id='16x', stride=1, bn=True)
        x = Dropout(0.2, name='drop')(x)
        x = Conv2D(1, (3, 3), activation='relu', name='OutputConv', strides=(1, 1), padding='same')(x)

        out = x

    model = Model(inputs=input_tensor, outputs=out, )

    return model


def load_depth_model(config):
    if config.loss_type == "HUBER":
        custom_object_dict = {'huber': huber}
    elif config.loss_type == "SIMSE":
        custom_object_dict = {'simse': simse_create}
    else:
        custom_object_dict = {}

    model = load_model(config.model_dir + config.model_name, custom_objects=custom_object_dict)

    return model


def depth_model_v2(config):

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

            return resnet_model.output

    input_tensor = Input(shape=config.input_size)
    x = resnet(input_tensor)
    conv = DeConv(1024, padding="valid", activation="relu", kernel_size=3)(x)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(512, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(128, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(32, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(8, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = UpSampling2D((2, 2))(conv)
    conv = DeConv(4, padding="valid", activation="relu", kernel_size=5)(conv)
    conv = DeConv(1, padding="valid", activation="relu", kernel_size=5)(conv)

    model = Model(inputs=input_tensor, outputs=conv)
    return model


def depth_model_v3(config):

    def up_project(input, size, id):

        with K.name_scope('up_project_' + id):

            up = UpSampling2D((2, 2),name='upsample_'+id)(input)

            branch1 = DeConv(size, padding="same", activation="relu", kernel_size=5, name='deconv1_b1_'+id)(up)
            branch1 = BatchNormalization(name='bn1_b1_'+id)(branch1)
            branch1 = DeConv(size, padding="same", activation=None, kernel_size=3, name='deconv2_b1_'+id)(branch1)
            branch1 = BatchNormalization(name='bn2_b1_'+id)(branch1)

            branch2 = DeConv(size, padding="same", activation=None, kernel_size=5, name='deconv1_b2_'+id)(up)
            branch2 = BatchNormalization( name='bn_out_'+id)(branch2)

            out = Add( name='add_'+id)([branch1, branch2])

            out = Activation('relu', name='relu_b1_'+id)(out)

            return out

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
        x = resnet(input_tensor)

        with K.name_scope('upscale'):

            x = DeConv(1024, (1, 1), activation=None, name='layer1', padding='same')(x)
            x = BatchNormalization(name='layer1_bn')(x)
            x = up_project(x, 512, '2x')
            x = up_project(x, 256, '4x')
            x = up_project(x, 128, '8x')

            out = DeConv(1, 3, activation='relu', padding='valid', name= 'deconv_output')(x)

        model = Model(inputs=input_tensor, outputs=out)

    return model
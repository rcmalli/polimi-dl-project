import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dropout,\
    ZeroPadding2D, Concatenate, Activation,Add
from tensorflow.keras.models import Model


# class CustomPadding2D(ZeroPadding2D):
#     def call(self, x, mask=None):
#         pattern = [[0, 0],
#                    [self.top_pad, self.bottom_pad],
#                    [self.left_pad, self.right_pad],
#                    [0, 0]]
#         return tf.pad(x, pattern)


def unpool_as_conv(size, input, id, stride=1, relu=False, bn=True):


    layerName = "layer%s_ConvA" % (id)

    outA = Conv2D(size[3], (3, 3), activation='relu', name=layerName, padding='same')(input)

    layerName = "layer%s_ConvB" % (id)
    outB = ZeroPadding2D(padding=((1, 0), (1, 1)))(input)
    outB = Conv2D(size[3], (2, 3), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(outB)

    layerName = "layer%s_ConvC" % (id)

    outC =  ZeroPadding2D(padding=((1, 1), (1, 0)))(input)
    outC = Conv2D(size[3], (3, 2), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(outC)

    layerName = "layer%s_ConvD" % (id)

    outD =  ZeroPadding2D(padding=((1, 0), (1, 0)))(input)
    outD = Conv2D(size[3], (2, 2), activation='relu', name=layerName, strides=(stride, stride), padding='valid')(outD)


    left = Concatenate(axis=1)([outA,outB])
    right = Concatenate(axis=1)([outC, outD])
    out = Concatenate(axis=2)([left, right])

    if bn:
        layerName = "layer%s_BN" % (id)
        out = BatchNormalization(name=layerName)(out)

    if relu:
        out = Activation('relu')(out)

    return out


def up_project(input, size, id, stride=1, bn=True):


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
    out = Add(name=layerName)([branch1_output,branch2_output])

    layerName = "layer%s_ReLU" % (id)
    out = Activation('relu', name=layerName)(out)


    return out

def resnet(input_tensor):

    resnet_model = tf.keras.applications.ResNet50(weights='imagenet',
                                                  include_top=False, input_tensor=input_tensor)


    return resnet_model.get_layer('activation_48').output


def depth_model():
    input_tensor = Input(shape=(224, 224, 3))
    x = resnet(input_tensor)

    x = Conv2D(1024, (1, 1), activation='relu', name='layer1', strides=(1, 1), padding='same')(x)
    x = BatchNormalization(name='layer1_bn')(x)
    x = up_project(x, [3, 3, 1024, 512], id='2x', stride=1, bn=True)
    x = up_project(x, [3, 3, 512, 256], id='4x', stride=1, bn=True)
    x = up_project(x, [3, 3, 256, 128], id='8x', stride=1, bn=True)
    x = up_project(x, [3, 3, 128, 64], id='16x', stride=1, bn=True)
    x = Dropout(0.2, name='drop')(x)
    x = Conv2D(1, (3, 3), activation='relu', name='OutputConv', strides=(1, 1), padding='same')(x)

    out = x

    model = Model(inputs=input_tensor, outputs=out)


    return model

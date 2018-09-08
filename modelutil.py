import numpy as np
import pandas as pd

from random import randint

import matplotlib.pyplot as plt

plt.style.use('seaborn-white')
import seaborn as sns

sns.set_style("white")

from sklearn.model_selection import train_test_split

from skimage.transform import resize

from keras.preprocessing.image import load_img
from keras import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout,BatchNormalization,Activation
import keras
from keras.losses import binary_crossentropy
from keras import backend as K
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
from keras.layers import Activation, Conv2D, MaxPooling2D, BatchNormalization, Input, DepthwiseConv2D, add, Dropout, \
    AveragePooling2D, Concatenate
from keras.models import Model
import keras.backend as K
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
from tqdm import tqdm_notebook
from keras.backend import common

img_size_ori = 101
img_size_target = 128

input_shape = (128, 128, 1)
out_stride = 16

class BilinearUpsampling(Layer):

    def __init__(self, upsampling=(2, 2), data_format=None, **kwargs):
        super(BilinearUpsampling, self).__init__(**kwargs)
        self.data_format = common.normalize_data_format(data_format)
        # self.data_format = conv_utils.normalize_padding(data_format)
        self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'size')
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        height = self.upsampling[0] * \
                 input_shape[1] if input_shape[1] is not None else None
        width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        return K.tf.image.resize_bilinear(inputs, (int(inputs.shape[1] * self.upsampling[0]),
                                                   int(inputs.shape[2] * self.upsampling[1])))

    def get_config(self):
        config = {'size': self.upsampling,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def xception_block(x, channels):
    ##separable conv1
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv2
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)

    ##separable conv3
    x = Activation("relu")(x)
    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Conv2D(channels, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    return x


def res_xception_block(x, channels):
    res = x
    x = xception_block(x, channels)
    x = add([x, res])
    return x

def upsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_target, img_size_target), mode='constant', preserve_range=True)
    # res = np.zeros((img_size_target, img_size_target), dtype=img.dtype)
    # res[:img_size_ori, :img_size_ori] = img
    # return res


def downsample(img):
    if img_size_ori == img_size_target:
        return img
    return resize(img, (img_size_ori, img_size_ori), mode='constant', preserve_range=True)
    # return img[:img_size_ori, :img_size_ori]


def aspp(x, input_shape, out_stride):
    b0 = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    b0 = BatchNormalization()(b0)
    b0 = Activation("relu")(b0)

    b1 = DepthwiseConv2D((3, 3), dilation_rate=(6, 6), padding="same", use_bias=False)(x)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)
    b1 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b1)
    b1 = BatchNormalization()(b1)
    b1 = Activation("relu")(b1)

    b2 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)
    b2 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b2)
    b2 = BatchNormalization()(b2)
    b2 = Activation("relu")(b2)

    b3 = DepthwiseConv2D((3, 3), dilation_rate=(12, 12), padding="same", use_bias=False)(x)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)
    b3 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b3)
    b3 = BatchNormalization()(b3)
    b3 = Activation("relu")(b3)

    out_shape = int(input_shape[0] / out_stride)
    b4 = AveragePooling2D(pool_size=(out_shape, out_shape))(x)
    b4 = Conv2D(256, (1, 1), padding="same", use_bias=False)(b4)
    b4 = BatchNormalization()(b4)
    b4 = Activation("relu")(b4)
    b4 = BilinearUpsampling((out_shape, out_shape))(b4)

    x = Concatenate()([b4, b0, b1, b2, b3])
    return x


def build_model(input_layer, start_neurons):
    # 128 -> 64
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(input_layer)
    conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    # conv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # 64 -> 32
    pool1 = Conv2D(start_neurons * 2, (1, 1),padding="same", use_bias=False)(pool1)
    res = Conv2D(start_neurons * 2, (1, 1), strides=(2, 2), padding="same", use_bias=False)(pool1)
    res = BatchNormalization()(res)
    pool1 = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(pool1)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    conv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(conv2)
    pool2 = add([conv2,res])
    # pool2 = MaxPooling2D((2, 2))(conv2)
    # pool2 = Dropout(0.5)(pool2)


    # 32 -> 16
    skip = BatchNormalization()(pool2)
    res3 = Conv2D(start_neurons * 4, (1, 1), strides=(2, 2), padding="same", use_bias=False)(pool2)
    res3 = BatchNormalization()(res3)
    pool2 = DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False)(pool2)
    conv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(pool2)
    conv3 = Conv2D(start_neurons * 4, (1, 1), activation="relu", padding="same")(conv3)
    conv3 = Conv2D(start_neurons * 4, (1, 1), activation="relu", padding="same")(conv3)

    conv3 = Activation('relu')(conv3)
    # pool3 = MaxPooling2D((2, 2))(conv3)
    # pool3 = Dropout(0.5)(pool3)
    pool3 = add([conv3,res3])

    # 16 -> 8
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(pool3)
    conv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(conv4)
    conv4 = Conv2D(start_neurons * 8, (1, 1), activation="relu", padding="same")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    x = MaxPooling2D((2, 2))(conv4)
    # pool4 = Dropout(0.5)(pool4)
    # for i in range(6):
    #     x = res_xception_block(x, 256)
    # aspp
    x = aspp(x, input_shape, out_stride)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = Dropout(0.9)(x)

    ##decoder
    x = BilinearUpsampling((4, 4))(x)
    dec_skip = Conv2D(256, (1, 1), padding="same", use_bias=False)(skip)
    dec_skip = BatchNormalization()(dec_skip)
    dec_skip = Activation("relu")(dec_skip)
    x = Concatenate()([x, dec_skip])

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = DepthwiseConv2D((3, 3), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(256, (1, 1), padding="same", use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = BilinearUpsampling((4, 4))(x)
    x = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)
    return x

    # Middle
    # convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(pool4)
    # convm = Conv2D(start_neurons * 16, (3, 3), activation="relu", padding="same")(convm)
    #
    # # 8 -> 16
    # deconv4 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(convm)
    # uconv4 = concatenate([deconv4, conv4])
    # # uconv4 = Dropout(0.5)(uconv4)
    # uconv4 = BatchNormalization()(uconv4)
    # uconv4 = Activation('relu')(uconv4)
    # uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    # uconv4 = Conv2D(start_neurons * 8, (3, 3), activation="relu", padding="same")(uconv4)
    #
    # # 16 -> 32
    # deconv3 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv4)
    # uconv3 = concatenate([deconv3, conv3])
    # # uconv3 = Dropout(0.5)(uconv3)
    # uconv3 = BatchNormalization()(uconv3)
    # uconv3 = Activation('relu')(uconv3)
    # uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    # uconv3 = Conv2D(start_neurons * 4, (3, 3), activation="relu", padding="same")(uconv3)
    #
    # # 32 -> 64
    # deconv2 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    # uconv2 = concatenate([deconv2, conv2])
    # # uconv2 = Dropout(0.5)(uconv2)
    # uconv2 = BatchNormalization()(uconv2)
    # uconv2 = Activation('relu')(uconv2)
    # uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    # uconv2 = Conv2D(start_neurons * 2, (3, 3), activation="relu", padding="same")(uconv2)
    #
    # # 64 -> 128
    # deconv1 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    # uconv1 = concatenate([deconv1, conv1])
    # # uconv1 = Dropout(0.5)(uconv1)
    # uconv1 = BatchNormalization()(uconv1)
    # uconv1 = Activation('relu')(uconv1)
    # uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    # uconv1 = Conv2D(start_neurons * 1, (3, 3), activation="relu", padding="same")(uconv1)
    #
    # # uconv1 = Dropout(0.5)(uconv1)
    # output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)
    #
    # return output_layer


def HourGlassNet(input_layer, start_neurons):
    output_layer = build_model(input_layer,32)
    output_layer = build_model(output_layer, 32)
    return output_layer

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

if __name__ == "__main__":
    input_layer = Input((img_size_target, img_size_target, 1))
    output_layer = HourGlassNet(input_layer,32)
    model = Model(input_layer, output_layer)

    model.compile(loss=bce_dice_loss, optimizer="adam", metrics=["accuracy"])

    model.summary()
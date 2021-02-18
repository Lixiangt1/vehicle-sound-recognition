import numpy as np
import tensorflow as tf
from tensorflow import keras

class BasicResdual(keras.layers.Layer):

    def __init__(self, num_filters, strides=1,downsample=None, **kwargs):
        super(BasicResdual, self).__init__(**kwargs)
        self.strides=strides
        self.conv1 = keras.layers.Conv2D(num_filters,
                                         kernel_size=3,
                                         strides=strides,
                                         padding='same',
                                         use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(num_filters,
                                         kernel_size=3,
                                         strides=1,
                                         padding='same',
                                         use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.add = keras.layers.Add()
        self.shortcut = keras.layers.Conv2D(num_filters,
                                            kernel_size=1,
                                            strides=strides,
                                            padding='same',
                                            use_bias=False)
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    def call(self, input):
        x = input
        y = self.conv1(x)
        y = self.bn1(y)
        y = keras.layers.Activation('relu')(y)
        y=self.conv2(y)
        y = self.bn2(y)
        y = keras.layers.Activation('relu')(y)
        if self.strides != 1:
            x = self.shortcut(x)
            x = self.bn3(x)
        y = self.add([x, y])
        y = keras.layers.Activation('relu')(y)
        return y


class BottleNeck(keras.layers.Layer):

    def __init__(self, num_filters, strides=1, downsample=None,**kwargs):
        super(BottleNeck, self).__init__(**kwargs)
        self.downsample = downsample
        self.conv1 = keras.layers.Conv2D(num_filters,
                                         kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         use_bias=False)
        self.bn1 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv2 = keras.layers.Conv2D(num_filters,
                                         kernel_size=3,
                                         strides=strides,
                                         padding='same',
                                         use_bias=False)
        self.bn2 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.conv3 = keras.layers.Conv2D(num_filters * 4,
                                         kernel_size=1,
                                         strides=1,
                                         padding='same',
                                         use_bias=False)
        self.bn3 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.shortcut = keras.layers.Conv2D(num_filters * 4,
                                            kernel_size=1,
                                            strides=strides,
                                            padding='same',
                                            use_bias=False)
        self.bn4 = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.add = keras.layers.Add()

    def call(self, input):
        x = input
        y = self.conv1(x)
        y = self.bn1(y)
        y = keras.layers.Activation('relu')(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = keras.layers.Activation('relu')(y)
        y = self.conv3(y)
        y = self.bn3(y)
        if self.downsample is not None:
            x = self.shortcut(x)
            x = self.bn4(x)
        y = self.add([x, y])
        y = keras.layers.Activation('relu')(y)
        return y


class ResdualBlock(keras.layers.Layer):
    def __init__(self, unit, num_filters, num_blocks, strides=1, first_block=False, **kwargs):
        super(ResdualBlock, self).__init__(**kwargs)
        self.layers=[]
        for i in range(num_blocks):
            if i==0 and not first_block:
                self.layers.append(unit(num_filters,downsample=1,strides=strides))
            elif i==0:
                self.layers.append(unit(num_filters,downsample=1))
            else:
                self.layers.append(unit(num_filters))

    def call(self, input):
        y = input
        for layer in self.layers:
            y = layer(y)
        return y


def resnet(unit, num_blocks, num_classes=1000, include_top=True):
    input_image = keras.layers.Input(shape=(224, 224, 1), dtype='float32')
    x = keras.layers.Conv2D(64,
                            kernel_size=7,
                            strides=2,
                            padding='same',
                            use_bias=False)(input_image)
    x = keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = keras.activations.relu(x)
    x = keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    x = ResdualBlock(unit, 64, num_blocks[0], first_block=True)(x)
    x = ResdualBlock(unit, 128, num_blocks[1], strides=2)(x)
    x = ResdualBlock(unit, 256, num_blocks[2], strides=2)(x)
    x = ResdualBlock(unit, 512, num_blocks[3], strides=2)(x)
    y = x
    if include_top:
        y=keras.layers.GlobalAveragePooling2D()(y)
        y=keras.layers.Dense(1024,activation='relu')(y)
        y=keras.layers.Dropout(0.4)(y)
        y=keras.layers.Dense(num_classes,activation='softmax')(y)

    model=keras.Model(inputs=input_image,outputs=y)
    return model

def resnet18(num_classes=1000,include_top=True):
    return resnet(BasicResdual,[2,2,2,2],num_classes,include_top)

def resnet34(num_classes=1000,include_top=True):
    return resnet(BasicResdual,[3,4,6,3],num_classes,include_top)

def resnet50(num_classes=1000,include_top=True):
    return resnet(BottleNeck,[3,4,6,3],num_classes,include_top)

def resnet101(num_classes=1000,include_top=True):
    return resnet(BottleNeck,[3,8,36,3],num_classes,include_top)


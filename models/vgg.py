
import numpy as np

#pip3 install tensorlayerx to install
# import tensorlayerx as tlx
# from tensorlayerx import logging
# from tensorlayerx.files import assign_weights, maybe_download_and_extract
# from tensorlayerx.nn import Module
# from tensorlayerx.nn import (BatchNorm, Conv2d, Linear, Flatten, Input, Sequential, MaxPool2d)
from tensorflow import Module, Variable, Tensor
from tensorflow.keras import layers, activations, Model
import tensorflow as tf
from tensorflow import keras

layer_names = [
    ['conv1_1', 'conv1_2'], 'pool1', ['conv2_1', 'conv2_2'], 'pool2',
    ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4'], 'pool3', ['conv4_1', 'conv4_2', 'conv4_3', 'conv4_4'], 'pool4',
    ['conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'], 'pool5', 'flatten', 'fc1_relu', 'fc2_relu', 'outputs'
]


class VGG19(Module):

    def __init__(self, batch_norm=False, end_with='outputs', name=None):
        super(VGG19, self).__init__(name=name)
        self.end_with = end_with

        config = [
            [64, 64], 'M', [128, 128], 'M', [256, 256, 256, 256], 'M', [512, 512, 512, 512], 'M', [512, 512, 512, 512],
            'M', 'F', 'fc1', 'fc2', 'O'
        ]
        self.make_layer = make_layers(config, batch_norm, end_with)

    def forward(self, inputs):
        """
        inputs : tensor
            Shape [None, 224, 224, 3], value range [0, 1].
        """

        inputs = inputs * 255. - tf.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32))
        out = self.make_layer(inputs)
        return out


def make_layers(config, batch_norm=False, end_with='outputs'):
    layer_list = []
    is_end = False
    for list_idx, list_group in enumerate(config):
        if type(list_group) is list:
            for layer_idx, size in enumerate(list_group):
                layer_name = layer_names[list_idx][layer_idx]
                layer_list.append(
                    layers.Conv2D(size, 3, strides=(1, 1), padding='same', activation=layers.ReLU())
                )
                if batch_norm:
                    layers.append(layers.BatchNormalization())
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[list_idx][layer_idx]
            if list_group == 'M':
                layer_list.append(layers.MaxPool2D(pool_size=(2, 2), stride=(2, 2), padding='same'))
            elif list_group == 'O':
                layer_list.append(layers.Dense(1000))
            elif list_group == 'F':
                layer_list.append(layers.Flatten())
            elif list_group == 'fc1':
                layer_list.append(layers.Dense(4096, activation=layers.ReLU()))
            elif list_group == 'fc2':
                layer_list.append(layers.Dense(4096, activation=layers.ReLU()))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return keras.Sequential(layer_list)


def vgg19(end_with='outputs', name=None):
    model = VGG19(batch_norm=False, end_with=end_with, name=name)
    return model

VGG19 = vgg19

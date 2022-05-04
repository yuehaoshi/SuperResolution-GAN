import enum
import os

import numpy as np
from sqlalchemy import true


#pip3 install tensorlayerx to install
import tensorlayerx as tlx
from tensorlayerx import logging
from tensorlayerx.files import assign_weights, maybe_download_and_extract
from tensorlayerx.nn import Module
from tensorlayerx.nn import (BatchNorm, Conv2d, Linear, Flatten, Input, Sequential, MaxPool2d)
# from tensorflow import Module, Variable, Tensor
# from tensorflow.keras import layers, activations, Model

__all__ = [
    'VGG',
    'vgg16',
    'vgg19',
    'VGG16',
    'VGG19',
]

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

        inputs = inputs * 255. - tlx.convert_to_tensor(np.array([123.68, 116.779, 103.939], dtype=np.float32))
        out = self.make_layer(inputs)
        return out


def make_layers(config, batch_norm=False, end_with='outputs'):
    layers = []
    is_end = False
    for list_idx, list_group in enumerate(config):
        if type(list_group) is list:
            for layer_idx, size in enumerate(list_group):
                layer_name = layer_names[list_idx][layer_idx]
                if layer_idx == 0:
                    if list_idx > 0:
                        input_size = config[list_idx-2][-1]
                    else:
                        input_size = 3
                else:
                    input_size = list_group[layer_idx-1]
                layers.append(
                    Conv2d(
                        out_channels=size, kernel_size=(3, 3), stride=(1, 1), act=tlx.ReLU, padding='SAME',
                        in_channels=input_size, name=layer_name
                    )
                )
                if batch_norm:
                    layers.append(BatchNorm(num_features=size))
                if layer_name == end_with:
                    is_end = True
                    break
        else:
            layer_name = layer_names[list_idx]
            if list_group == 'M':
                layers.append(MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding='SAME', name=layer_name))
            elif list_group == 'O':
                layers.append(Linear(out_features=1000, in_features=4096, name=layer_name))
            elif list_group == 'F':
                layers.append(Flatten(name='flatten'))
            elif list_group == 'fc1':
                layers.append(Linear(out_features=4096, act=tlx.ReLU, in_features=512 * 7 * 7, name=layer_name))
            elif list_group == 'fc2':
                layers.append(Linear(out_features=4096, act=tlx.ReLU, in_features=4096, name=layer_name))
            if layer_name == end_with:
                is_end = True
        if is_end:
            break
    return Sequential(layers)


def vgg19(pretrained=False, end_with='outputs', mode='dynamic', name=None):
    """Pre-trained VGG19 model.
    Parameters
    ------------
    pretrained : boolean
        Whether to load pretrained weights. Default False.
    end_with : str
        The end point of the model. Default ``fc3_relu`` i.e. the whole model.
    mode : str.
        Model building mode, 'dynamic' or 'static'. Default 'dynamic'.
    name : None or str
        A unique layer name.
    Examples
    ---------
    Classify ImageNet classes with VGG19, see `tutorial_models_vgg.py <https://github.com/tensorlayer/TensorLayerX/blob/main/examples/model_zoo/vgg.py>`__
    With TensorLayerx
    >>> # get the whole model, without pre-trained VGG parameters
    >>> vgg = vgg19()
    >>> # get the whole model, restore pre-trained VGG parameters
    >>> vgg = vgg19(pretrained=True)
    >>> # use for inferencing
    >>> output = vgg(img)
    >>> probs = tlx.ops.softmax(output)[0].numpy()
    """
    if mode == 'dynamic':
        model = VGG19(batch_norm=False, end_with=end_with, name=name)
    elif mode == 'static':
        raise NotImplementedError
    else:
        raise Exception("No such mode %s" % mode)
    return model

VGG19 = vgg19
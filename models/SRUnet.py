import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model, layers, losses, activations



class Encoder(layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def forward(self, x:Tensor):
        return x

class SRUnet(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1 = layers.Conv2D()
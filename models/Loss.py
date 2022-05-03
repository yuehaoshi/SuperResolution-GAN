from turtle import forward
from tensorflow import losses, Tensor
from tensorflow.keras import layers, Model


class MSE_loss(layers.Layer):
    """
    Warning!! MSE_loss not used

    """

    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

    def call(self, x1:Tensor, x2, Tensor) -> Tensor:
        return losses.mse(x1, x2)
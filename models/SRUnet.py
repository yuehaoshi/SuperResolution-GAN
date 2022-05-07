from typing import List
import tensorflow as tf
from tensorflow import Tensor
from tensorflow.keras import Model, layers, losses, activations
import time


class ResBlock(layers.Layer):
    def __init__(self, filters:int, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.conv1 = layers.Conv2D(filters, 3, padding="same")
        self.conv2 = layers.Conv2D(filters, 3, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, x:Tensor):
        y = self.conv1(x)
        y = self.bn1(y)
        y = activations.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = activations.relu(y)
        
        return x + y

class SubPixConv2D(layers.Layer):
    def __init__(self, filters: int, scale: int = 2, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.scale = scale
        self.conv1 = layers.Conv2D(filters*scale*2, 3, padding="same")
        self.bn = layers.BatchNormalization()

    def call(self, x: Tensor):
        y = self.conv1(x)
        y = tf.nn.depth_to_space(y, self.scale)
        y = self.bn(y)
        y = activations.relu(y)
        return y

class UpBlock(layers.Layer):
    def __init__(self, filters:List, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.up = SubPixConv2D(filters[0])
        self.filters:List = filters
        self.conv1 = layers.Conv2D(filters[0], 3, padding="same")
        self.conv2 = layers.Conv2D(filters[1], 1, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    def call(self, x_LR: Tensor, x_HR:Tensor):
        y = self.up(x_LR)
        y = tf.concat([y, x_HR], axis=3)
        y = self.conv1(y)
        y = self.bn1(y)
        y = activations.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y = activations.relu(y)
        return y

class SRUnet(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.b1_1 = ResBlock(64)
        self.b1_2 = ResBlock(64)
        self.b1_3 = ResBlock(64)

        self.b2_1 = ResBlock(128)
        self.b2_2 = ResBlock(128)
        self.b2_3 = ResBlock(128)

        self.b3_1 = ResBlock(256)
        self.b3_2 = ResBlock(256)
        self.b3_3 = ResBlock(256)
        self.b3_4 = ResBlock(256)
        self.b3_5 = ResBlock(256)

        self.b4_1 = ResBlock(512)
        self.b4_2 = ResBlock(512)
        self.b4_3 = ResBlock(512)
        self.b4_4 = ResBlock(512)

        self.mp1 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.mp2 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")
        self.mp3 = layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding="same")

        self.up1 = UpBlock([256, 128])
        self.up2 = UpBlock([128, 64])
        self.up3 = UpBlock([64, 32])
        self.sr1 = SubPixConv2D(8)
        self.sr2 = SubPixConv2D(3)

        self.conv0 = layers.Conv2D(64, 3, padding="same", name="sdf")
        self.conv1 = layers.Conv2D(128, 3, padding="same")
        self.conv2 = layers.Conv2D(256, 3, padding="same")
        self.conv3 = layers.Conv2D(512, 3, padding="same", name="las_conv")

    def call(self, x:Tensor):
        y = self.conv0(x)
        y = self.b1_1(y)
        y = self.b1_2(y)
        y = self.b1_3(y)
        y1 = self.conv1(y)
        x2 = self.mp1(y1)

        y = self.b2_1(x2)
        y = self.b2_2(y)
        y = self.b2_3(y)
        y2 = self.conv2(y)
        x4 = self.mp2(y2)

        y = self.b3_1(x4)
        y = self.b3_2(y)
        y = self.b3_3(y)
        y = self.b3_4(y)
        y = self.b3_5(y)
        y3 = self.conv3(y)
        x8 = self.mp3(y3)

        y = self.b4_1(x8)
        y = self.b4_2(y)
        y = self.b4_3(y)
        y4 = self.b4_4(y)

        y = self.up1(y4, y3)
        y = self.up2(y, y2)
        y = self.up3(y, y1)
        y = tf.concat([y, x], axis=3)
        y = self.sr1(y)
        y = self.sr2(y)

        return y



def test():
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass
    imsize = (1, 512, 512, 3)
    tf.random.set_seed(0)
    x = tf.random.uniform(imsize)
    sr = SRUnet()

    sr.build(imsize)
    sr.summary()
    with tf.GradientTape() as tape:
        y:Tensor = sr(x)

    grads = tape.gradient(y, sr.trainable_weights)
    # print(grads)
    print(y.shape)

    time.sleep(10)
    # logging.basicConfig(filename='SRResnet.log', encoding='utf-8', level=logging.DEBUG, filemode='a+')
    # logging.info(str(y))


if __name__ == "__main__":
    test()
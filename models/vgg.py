import tensorflow.keras.applications.vgg19 as vgg

from tensorflow.keras import Model


def VGG() -> Model:
    base_model = vgg.VGG19(False)
    VGG = Model(inputs=base_model.input,
                outputs=base_model.get_layer("block3_conv3").output)
    return VGG

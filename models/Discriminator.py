from tensorflow.keras import layers, activations


class Discriminator(layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.prelu = layers.PReLU()
        self.conv1 = layers.Conv2D(64, 3, strides=(1, 1), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.conv2 = layers.Conv2D(64, 3, strides=(2, 2), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.bn1 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(128, 3, strides=(1, 1), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.bn2 = layers.BatchNormalization()
        self.conv4 = layers.Conv2D(128, 3, strides=(2, 2), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.bn3 = layers.BatchNormalization()
        self.conv5 = layers.Conv2D(256, 3, strides=(1, 1), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.bn4 = layers.BatchNormalization()
        self.conv6 = layers.Conv2D(256, 3, strides=(2, 2), padding='same', activation=layers.LeakyReLU(alpha=0.18))
        self.bn5 = layers.BatchNormalization()
        self.flat = layers.Flatten()
        self.dense1 = layers.Dense(512, activation=layers.LeakyReLU(alpha=0.18))
        self.dense2 = layers.Dense(1)
        self.sigmoid = activations.sigmoid()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        result = x
        x = self.sigmoid(x)
        return x, result


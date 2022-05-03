import tensorflow as tf
from Discriminator import Discriminator
import config
from models.SRResnet import SRResnet
from tensorflow import losses, Tensor
from tensorflow.keras import Model
import Dataloader
from tqdm import tqdm


@tf.function
def discriminator_loss(fake: Tensor, real: Tensor, discriminator: Model):
    fake = discriminator(fake)
    real = discriminator(real)
    loss = losses.binary_crossentropy(tf.zeros_like(
        fake), fake) + losses.binary_crossentropy(tf.ones_like(real), real)
    return loss


@tf.function
def generator_loss(fake: Tensor, discriminator: Model):
    pred = discriminator(fake)
    loss = losses.binary_crossentropy(tf.ones_like(pred), pred)
    return loss


def train_mse():
    optimizer_SR = tf.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, beta_1=0.9)
    optimizer_D = tf.optimizers.Adam(
        learning_rate=config.LEARNING_RATE, beta_1=0.9)
    model = SRResnet(B=16)
    discriminator = Discriminator()

    for ep in tqdm(range(config.EPOCHS)):
        for step, (X_train, y_train) in enumerate(Dataloader):

            # update discriminator
            output = model(X_train, training=False)
            with tf.GradientTape() as tape:
                loss = discriminator_loss(output, y_train, discriminator)
            grads = tape.gradient(loss, discriminator.trainable_weights)
            optimizer_D.apply_gradients(
                zip(grads, discriminator.trainable_weights))

            # update SRResnet (generator)
            with tf.GradientTape() as tape:
                output = model(X_train, training=True)
                loss = losses.mse(
                    output, y_train) + config.DISCRIMINATOR_WEIGHT * generator_loss(output, discriminator)
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer_SR.apply_gradients(zip(grads, model.trainable_weights))


if __name__ == "__main__":
    train_mse()

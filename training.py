import tensorflow as tf
from models.Discriminator import Discriminator
import config
from models.SRResnet import SRResnet
from tensorflow import losses, Tensor, keras
from tensorflow.keras import Model, layers
from Dataloader import DIV2KDataset
from tqdm import tqdm


@tf.function
def discriminator_loss(fake: Tensor, real: Tensor, discriminator: Model):
    fake = discriminator(fake)
    real = discriminator(real)
    print(fake.shape)
    print(real.shape)
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
#     discriminator = Discriminator()
    reference_model = keras.applications.resnet50.ResNet50(
                        include_top=False,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling='avg'
                    )
    result = reference_model.output
    result = layers.Dense(1)(result)
    discriminator = Model(inputs=reference_model.input, outputs=result)
    dataset = DIV2KDataset(config.DIV2K_PATH, 'train')
    train_data = tf.data.Dataset.from_generator(
        dataset.pair_generator,
        output_signature=(tf.TensorSpec((None, None, 3)), tf.TensorSpec((None, None, 3))))
    # train_data = train_data.batch(config.BATCH_SIZE) # cannot batch because of different image size
    train_data = train_data.shuffle(32)
    for ep in tqdm(range(config.EPOCHS)):

        for step, (X_train, y_train) in train_data.enumerate():
            # batch size of 1
            X_train:Tensor = tf.expand_dims(X_train, 0)
            y_train:Tensor = tf.expand_dims(y_train, 0)

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

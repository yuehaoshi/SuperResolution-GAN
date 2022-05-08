import tensorflow as tf
from models.Discriminator import Discriminator
import config
from models.SRUnet import SRUnet
from tensorflow import losses, Tensor, keras
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import resnet50
from Dataloader import DIV2KDataset
from tqdm import tqdm
import datetime
from models.vgg import VGG


def discriminator_loss(fake: Tensor, real: Tensor, discriminator: Model):
    fake = discriminator(resnet50.preprocess_input(fake))
    real = discriminator(resnet50.preprocess_input(real))
    # print(fake.shape)
    # print(real.shape)
    loss = losses.binary_crossentropy(tf.zeros_like(
        fake), fake) + losses.binary_crossentropy(tf.ones_like(real), real)
    return loss


def generator_loss(fake: Tensor, discriminator: Model):
    pred = discriminator(resnet50.preprocess_input(fake), training=False)
    loss = losses.binary_crossentropy(tf.ones_like(pred), pred)
    return loss


def train_mse(load_model=False):

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/unet' + current_time + '/train'
    test_log_dir = 'logs/unet' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # setup optimizers
    optimizer_SR = tf.optimizers.Adam(
        learning_rate=1e-4, beta_1=0.9)
    optimizer_D = tf.optimizers.Adam(
        learning_rate=1e-4, beta_1=0.9)

    model = SRUnet()
    discriminator = Discriminator()
    if load_model:
        model.load_weights("checkpoints/unet-ep19/")
        discriminator.load_weights("checkpoints/resnet-ep19/")


    # vgg = VGG()

    dataset = DIV2KDataset(config.DIV2K_PATH, 'train', patches=8)
    train_data = tf.data.Dataset.from_generator(
        dataset.patch_generator,
        output_signature=(tf.TensorSpec((None, None, 3)), tf.TensorSpec((None, None, 3))))
    # train_data = train_data.batch(config.BATCH_SIZE) # cannot batch because of different image size
    train_data = train_data.shuffle(32)
    train_data = train_data.batch(4)
    prog = tqdm(range(config.EPOCHS))
    for ep in prog:
        en = train_data.enumerate()
        for step, (X_train, y_train) in en:
            # batch size of 1
            # y_train = tf.image.resize_with_pad(
            #     y_train, 1312, 1312, antialias=True)
            # X_train = tf.image.resize_with_pad(
            #     X_train, 328, 328, antialias=True)
            # X_train: Tensor = tf.expand_dims(X_train, 0)
            # y_train: Tensor = tf.expand_dims(y_train, 0)
            # print(X_train.shape)
            # print(y_train.shape)
            # update discriminator
            X_train = X_train / 255.
            y_train = y_train / 255.

            output = model(X_train, training=False)
            # print(output.shape)
            with tf.GradientTape() as tape:
                loss = discriminator_loss(output, y_train, discriminator)
            grads = tape.gradient(loss, discriminator.trainable_weights)
            optimizer_D.apply_gradients(
                zip(grads, discriminator.trainable_weights))

            # update SRResnet (generator)
            with tf.GradientTape() as tape:
                output = model(X_train, training=True)
                # MSE loss
                loss = tf.reduce_mean(losses.mse(
                    output, y_train)) + config.DISCRIMINATOR_WEIGHT * generator_loss(output, discriminator)
                # perceptual loss
                # loss = tf.reduce_mean(losses.mse(
                #     vgg(output), vgg(y_train))) + config.DISCRIMINATOR_WEIGHT * generator_loss(output, discriminator)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer_SR.apply_gradients(zip(grads, model.trainable_weights))
            if int(step) % 100 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar("loss", float(tf.reduce_mean(loss)), step=ep*800 + int(step))
                    tf.summary.image('gen image', output, step=ep*800 + int(step))
            prog.set_postfix({"step": int(step), "loss": float(tf.reduce_mean(loss))})
        
        model.save_weights(f"checkpoints/unet-ep{ep}/")
        discriminator.save_weights(f"checkpoints/resnet-ep{ep}/")


if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
    # Invalid device or cannot modify virtual devices once initialized.
        pass

    train_mse(False)

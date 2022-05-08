from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import Tensor
from Dataloader import DIV2KDataset
import config
from tqdm import tqdm


physical_devices = tf.config.list_physical_devices('GPU')
model = tf.keras.models.load_model("checkpoints/model-ep40.pth/")

dataset = DIV2KDataset(config.DIV2K_PATH, 'valid')
train_data = tf.data.Dataset.from_generator(
    dataset.pair_generator,
    output_signature=(tf.TensorSpec((None, None, 3)), tf.TensorSpec((None, None, 3))))
# train_data = train_data.batch(config.BATCH_SIZE) # cannot batch because of different image size
en = tqdm(train_data.enumerate())
for step, (X_train, y_train) in en:


    # batch size of 1
    X_train:Tensor = tf.expand_dims(X_train, 0)
    H, W, _ = X_train.shape
    H, W = H + 8 - (H % 8), W + 8 - (W % 8)
    X_train = tf.image.pad_to_bounding_box(X_train,0,0, H, W)
    output = model(X_train, training=False)
    output = tf.clip_by_value(output[0]/255, 0, 1)
    plt.imsave(f"outputs/{step}.png", output.numpy())
   



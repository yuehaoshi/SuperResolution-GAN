from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import Tensor
from Dataloader import DIV2KDataset
import config
from tqdm import tqdm
from models.SRResnet import SRResnet



physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
# Invalid device or cannot modify virtual devices once initialized.
    pass

model = tf.keras.models.load_model("checkpoints/model-ep40.pth/")

dataset = DIV2KDataset(config.DIV2K_PATH, 'train')
train_data = tf.data.Dataset.from_generator(
    dataset.pair_generator,
    output_signature=(tf.TensorSpec((None, None, 3)), tf.TensorSpec((None, None, 3))))
# train_data = train_data.batch(config.BATCH_SIZE) # cannot batch because of different image size
train_data = train_data.shuffle(2)
plt.figure()
prog = tqdm(range(config.EPOCHS))
en = train_data.enumerate()
for step, (X_train, y_train) in en:


    #     # batch size of 1
    X_train:Tensor = tf.expand_dims(X_train, 0)
        
    output = model(X_train, training=False)
    output = tf.clip_by_value(output[0]/255, 0, 1)
    plt.imsave(f"outputs/{step}.png", output.numpy())
    prog.set_postfix({"step":step})
   



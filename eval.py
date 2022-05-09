import tensorflow as tf
from tensorflow import Tensor
from Dataloader import DIV2KDataset
import config
from tqdm import tqdm
from PIL import Image
import numpy as np
from models.SRResnet import SRResnet

physical_devices = tf.config.list_physical_devices('GPU')

dataset = DIV2KDataset(config.DIV2K_PATH, 'valid', 'bicubic')
train_data = tf.data.Dataset.from_generator(
    dataset.pair_generator,
    output_signature=(tf.TensorSpec((None, None, 3)), tf.TensorSpec((None, None, 3))))
# train_data = train_data.batch(config.BATCH_SIZE) # cannot batch because of different image size

def eval_srunet():
    model = tf.keras.models.load_model("checkpoints/unet-ep19/")
    en = tqdm(train_data.enumerate())
    for step, (X_train, y_train) in en:


        # batch size of 1
        H, W, _ = X_train.shape
        
        X_train = tf.image.resize_with_crop_or_pad(X_train,328, 328)
        X_train:Tensor = tf.expand_dims(X_train, 0)
        output = model(X_train, training=False)
        output = tf.clip_by_value(output[0], 0, 255).numpy().astype(np.uint8)
        img = Image.fromarray(output)
        img.save(f"outputs/srunet/{step}.png")
   
def eval_srresnet():
    model = SRResnet()
    model.build((1,3,3,3))
    model.load_weights("checkpoints/srresnet/")
    
    en = tqdm(train_data.enumerate())
    ssim_list = []
    psnr_list = []
    ssim = 0
    psnr = 0
    for step, (X_train, y_train) in en:


        # batch size of 1
        H, W, _ = X_train.shape

        X_train:Tensor = tf.expand_dims(X_train, 0)
        output = model(X_train, training=False)
        output = tf.clip_by_value(output[0], 0, 255)
        s = tf.image.ssim(output, y_train, max_val=255, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03)
        p = tf.image.psnr(output, y_train, max_val=255)
        ssim += s
        psnr += p
        # print(ssim, psnr)
        # img = Image.fromarray(output)
        # img.save(f"outputs/srresnet/{step}.png")
    # print(ssim_list)
    # print(psnr_list)

eval_srresnet()
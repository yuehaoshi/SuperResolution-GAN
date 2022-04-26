import tensorflow as tf
import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Read the data using the tf.io.read_file function and extract the label from the path, returning (image, label) pairs
np.set_printoptions(precision=4)

flowers_root = tf.keras.utils.get_file(
    #Take 'flower_photos' as an example. Reference: https://www.tensorflow.org/guide/data
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True)
flowers_root = pathlib.Path(flowers_root)

for item in flowers_root.glob("*"):
    print(item.name)

list_ds = tf.data.Dataset.list_files(str(flowers_root/'*/*'))

for f in list_ds.take(5):
    print(f.numpy())


def process_path(file_path):
    label = tf.strings.split(file_path, os.sep)[-2]
    return tf.io.read_file(file_path), label


labeled_ds = list_ds.map(process_path)

for image_raw, label_text in labeled_ds.take(1):
    print(repr(image_raw.numpy()[:100]))
    print()
    print(label_text.numpy())
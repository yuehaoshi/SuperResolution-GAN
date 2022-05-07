import os
import numpy as np
from PIL import Image
from tensorflow import convert_to_tensor, Tensor, image

# adapted from https://github.com/krasserm/super-resolution/blob/master/data.py


class DIV2KDataset:
    def __init__(self, path, subset='train', downgrade="bicubic", scale=4,
                 repeat=True, image_ids=None, cache_images=True):
        self.path = path
        self.subset = subset
        self.cache_images = cache_images
        self.cache = {}
        self.downgrade = downgrade
        self.scale = scale

        if image_ids is None:
            if subset == 'train':
                self.image_ids = range(1, 801)
            else:
                self.image_ids = range(801, 901)
        else:
            self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def pair_generator(self):

        for id in self.image_ids:
            hr_path = os.path.join(
                self.path, f'DIV2K_{self.subset}_HR', f'{id:04}.png')
            lr_path = os.path.join(
                self.path, f'DIV2K_{self.subset}_LR_{self.downgrade}',
                f'X{self.scale}', f'{id:04}x{self.scale}.png')

            hr_img = self._image(hr_path)
            lr_img = self._image(lr_path)
            hr_img = image.resize_with_pad(hr_img, 1200, 1200, antialias=True)
            lr_img = image.resize_with_pad(lr_img, 300, 300, antialias=True)
            #hr_img = image.resize(hr_img, [1200, 1200], antialias=True)
            #lr_img = image.resize(lr_img, [300, 300], antialias=True)

            yield lr_img, hr_img

    def _image(self, path) -> Tensor:
        img = self.cache.get(path)
        if not img:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.cache_images:
                self.cache[path] = img
        return convert_to_tensor(img)

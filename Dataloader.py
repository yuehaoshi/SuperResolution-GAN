import os
import numpy as np
from PIL import Image


#adapted from https://github.com/krasserm/super-resolution/blob/master/data.py

class DIV2KDataset:
    def __init__(self, path, subset='train', image_ids=None, cache_images=True):
        self.path = path
        self.subset = subset
        self.cache_images = cache_images
        self.cache = {}

        if image_ids is None:
            if subset == 'train':
                self.image_ids = range(1, 801)
            else:
                self.image_ids = range(801, 901)
        else:
            self.image_ids = image_ids

    def __len__(self):
        return len(self.image_ids)

    def pair_generator(self, downgrade='bicubic', scale=2, repeat=True):
        while True:
            for id in self.image_ids:
                hr_path = os.path.join(self.path, f'DIV2K_{self.subset}_HR', f'{id:04}.png')
                lr_path = os.path.join(self.path, f'DIV2K_{self.subset}_LR_{downgrade}', f'X{scale}', f'{id:04}x{scale}.png')

                hr_img = self._image(hr_path)
                lr_img = self._image(lr_path)

                yield lr_img, hr_img

            if not repeat:
                break

    def _image(self, path):
        img = self.cache.get(path)
        if not img:
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.cache_images:
                self.cache[path] = img
        return img

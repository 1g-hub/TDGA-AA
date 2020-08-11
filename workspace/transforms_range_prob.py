# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random
import torchvision.transforms as transforms

from abc import ABC, abstractmethod
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


class BaseTransform(ABC):
    maxval = 0
    minval = 0

    def __init__(self, prob, mag):
        self.prob = prob
        self.mag = mag
        self.val = (mag / 30) * (self.maxval - self.minval) + self.minval

    def __call__(self, img):
        return transforms.RandomApply([self.transform], self.prob)(img)

    def __repr__(self):
        return '%s(prob=%.2f, magnitude=%.2f, val=%.2f)' % \
               (self.__class__.__name__, self.prob, self.mag, self.val)

    @abstractmethod
    def transform(self, img):
        pass


class ShearX(BaseTransform):  # [-0.3, 0.3]
    minval = 0
    maxval = 0.3

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, self.val, 0, 0, 1, 0))


class ShearY(BaseTransform):  # [-0.3, 0.3]
    minval = 0
    maxval = 0.3

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, self.val, 1, 0))


class TranslateX(BaseTransform):  # [-150, 150] => percentage: [-0.45, 0.45]
    minval = 0
    maxval = 0.33

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        self.val = self.val * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, self.val, 0, 1, 0))


class TranslateXabs(BaseTransform):  # [-150, 150] => percentage: [-0.45, 0.45]
    minval = 0
    maxval = 100

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, self.val, 0, 1, 0))


class TranslateY(BaseTransform):  # [-150, 150] => percentage: [-0.45, 0.45]
    minval = 0
    maxval = 0.33

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        self.val = self.val * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, self.val))


class TranslateYabs(BaseTransform):  # [-150, 150] => percentage: [-0.45, 0.45]
    minval = 0
    maxval = 100

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, self.val))


class Rotate(BaseTransform):  # [-30, 30]
    minval = 0
    maxval = 30

    def transform(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.rotate(self.val)


class AutoContrast(BaseTransform):
    maxval = 0
    minval = 0

    def transform(self, img):
        return PIL.ImageOps.autocontrast(img)


class Invert(BaseTransform):
    maxval = 0
    minval = 0

    def transform(self, img):
        return PIL.ImageOps.invert(img)


class Equalize(BaseTransform):
    maxval = 0
    minval = 0

    def transform(self, img):
        return PIL.ImageOps.equalize(img)


class Flip(BaseTransform):
    maxval = 0
    minval = 0

    def transform(self, img):
        return PIL.ImageOps.mirror(img)


class Solarize(BaseTransform):  # [0, 256]
    minval = 0
    maxval = 256

    def transform(self, img):
        return PIL.ImageOps.solarize(img, self.val)


class SolarizeAdd(BaseTransform):  # [0, 110]
    minval = 0
    maxval = 110

    def transform(self, img, threshold=128):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + self.val
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold)


class Posterize(BaseTransform):  # [4, 8]
    minval = 0
    maxval = 4

    def transform(self, img, addition=0, threshold=128):
        v = int(self.val)
        v = max(1, v)
        return PIL.ImageOps.posterize(img, v)


class Contrast(BaseTransform):  # [0.1,1.9]
    minval = 0.1
    maxval = 1.9

    def transform(self, img):
        return PIL.ImageEnhance.Contrast(img).enhance(self.val)


class Color(BaseTransform):  # [0.1,1.9]
    minval = 0.1
    maxval = 1.9

    def transform(self, img):
        return PIL.ImageEnhance.Color(img).enhance(self.val)


class Brightness(BaseTransform):  # [0.1,1.9]
    minval = 0.1
    maxval = 1.9

    def transform(self, img):
        return PIL.ImageEnhance.Brightness(img).enhance(self.val)


class Sharpness(BaseTransform):  # [0.1,1.9]
    minval = 0.1
    maxval = 1.9

    def transform(self, img):
        return PIL.ImageEnhance.Sharpness(img).enhance(self.val)


class Cutout(BaseTransform):  # [0, 60] => percentage: [0, 0.2]
    minval = 0
    maxval = 0.2

    def transform(self, img):
        if self.val <= 0.:
            return img
        v = self.val * img.size[0]
        return CutoutAbs(self.prob, v)(img)


class CutoutAbs(BaseTransform):  # [0, 60] => percentage: [0, 0.2]
    minval = 0
    maxval = 40

    def transform(self, img):
        if self.val < 0:
            return img
        w, h = img.size
        x0 = np.random.uniform(w)
        y0 = np.random.uniform(h)

        x0 = int(max(0, x0 - self.val / 2.))
        y0 = int(max(0, y0 - self.val / 2.))
        x1 = min(w, x0 + self.val)
        y1 = min(h, y0 + self.val)

        xy = (x0, y0, x1, y1)
        color = (125, 123, 114)
        # color = (0, 0, 0)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img


class Identity(BaseTransform):
    minval = 0
    maxval = 0

    def transform(self, img):
        return img


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # l = [
    #     Identity,
    #     ShearX,  # 0
    #     ShearY,  # 1
    #     TranslateX,  # 2
    #     TranslateY,  # 3
    #     Rotate,  # 4
    #     AutoContrast,  # 5
    #     # Invert,  # 6
    #     Equalize,  # 7
    #     Solarize,  # 8
    #     Posterize,  # 9
    #     Contrast,  # 10
    #     Color,  # 11
    #     Brightness,  # 12
    #     Sharpness,  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        Identity,
        AutoContrast,
        Equalize,
        Invert,
        Rotate,
        Posterize,
        Solarize,
        SolarizeAdd,
        Color,
        Contrast,
        Brightness,
        Sharpness,
        ShearX,
        ShearY,
        CutoutAbs,
        TranslateXabs,
        TranslateYabs,
    ]

    return l


class CutoutDefault(object):
    """
    Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    """
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img

    def __str__(self):
        return "CutoutDefault(length={})".format(self.length)


class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op in ops:
            img = op(self.m)(img)

        return img

    def __str__(self):
        return "RandAugment(N={}, M={})".format(self.n, self.m)


if __name__ == '__main__':
    l = augment_list()
    # l = [CutoutDefault]
    path = "horse.jpg"
    import os
    os.makedirs("transform_test/", exist_ok=True)
    img = Image.open(path).convert("RGB")

    for op in l:
        for mag in [0, 15, 30]:
            img_transformed = op(prob=1, mag=mag)(img)
            img_transformed.save("transform_test/transformed_{}_{}.png".format(str(op), mag))

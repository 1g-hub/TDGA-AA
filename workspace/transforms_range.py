# code in this file is adpated from rpmcruz/autoaugment
# https://github.com/rpmcruz/autoaugment/blob/master/transformations.py
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image


class ShearX:  # [-0.3, 0.3]
    def __init__(self, mag):
        minval = 0
        maxval = 0.3
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, self.val, 0, 0, 1, 0))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class ShearY:  # [-0.3, 0.3]
    def __init__(self, mag):
        minval = 0
        maxval = 0.3
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, self.val, 1, 0))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class TranslateX:  # [-150, 150] => percentage: [-0.45, 0.45]
    def __init__(self, mag):
        minval = 0
        maxval = 0.33
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        v = self.val * img.size[0]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class TranslateXabs:  # [-150, 150] => percentage: [-0.45, 0.45]
    def __init__(self, mag):
        minval = 0
        maxval = 100
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, self.val, 0, 1, 0))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class TranslateY:  # [-150, 150] => percentage: [-0.45, 0.45]
    def __init__(self, mag):
        minval = 0
        maxval = 0.33
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        v = self.val * img.size[1]
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class TranslateYabs:  # [-150, 150] => percentage: [-0.45, 0.45]
    def __init__(self, mag):
        minval = 0
        maxval = 100
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, self.val))

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Rotate:  # [-30, 30]
    def __init__(self, mag):
        minval = 0
        maxval = 30
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if random.random() > 0.5:
            self.val = -self.val
        return img.rotate(self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class AutoContrast:
    def __init__(self, mag):
        pass

    def __call__(self, img):
        return PIL.ImageOps.autocontrast(img)

    def __repr__(self):
        return '%s' % \
               (self.__class__.__name__,)


class Invert:
    def __init__(self, mag):
        pass

    def __call__(self, img):
        return PIL.ImageOps.invert(img)

    def __repr__(self):
        return '%s' % \
               (self.__class__.__name__,)


class Equalize:
    def __init__(self, mag):
        pass

    def __call__(self, img):
        return PIL.ImageOps.equalize(img)

    def __repr__(self):
        return '%s' % \
               (self.__class__.__name__,)


class Flip:
    def __init__(self, mag):
        pass

    def __call__(self, img):
        return PIL.ImageOps.mirror(img)

    def __repr__(self):
        return '%s' % \
               (self.__class__.__name__,)


class Solarize:  # [0, 256]
    def __init__(self, mag):
        minval = 0
        maxval = 256
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        return PIL.ImageOps.solarize(img, self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class SolarizeAdd:  # [0, 110]
    def __init__(self, mag):
        minval = 0
        maxval = 110
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img, threshold=128):
        img_np = np.array(img).astype(np.int)
        img_np = img_np + self.val
        img_np = np.clip(img_np, 0, 255)
        img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        return PIL.ImageOps.solarize(img, threshold)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Posterize:  # [4, 8]
    def __init__(self, mag):
        minval = 0
        maxval = 4
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img, addition=0, threshold=128):
        v = int(self.val)
        v = max(1, v)
        return PIL.ImageOps.posterize(img, v)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Contrast:  # [0.1,1.9]
    def __init__(self, mag):
        minval = 0.1
        maxval = 1.9
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        return PIL.ImageEnhance.Contrast(img).enhance(self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Color:  # [0.1,1.9]
    def __init__(self, mag):
        minval = 0.1
        maxval = 1.9
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        return PIL.ImageEnhance.Color(img).enhance(self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Brightness:  # [0.1,1.9]
    def __init__(self, mag):
        minval = 0.1
        maxval = 1.9
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        return PIL.ImageEnhance.Brightness(img).enhance(self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Sharpness:  # [0.1,1.9]
    def __init__(self, mag):
        minval = 0.1
        maxval = 1.9
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        return PIL.ImageEnhance.Sharpness(img).enhance(self.val)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class Cutout:  # [0, 60] => percentage: [0, 0.2]
    def __init__(self, mag):
        minval = 0
        maxval = 0.2
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
        if self.val <= 0.:
            return img
        v = self.val * img.size[0]
        return CutoutAbs(v)(img)

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


class CutoutAbs:  # [0, 60] => percentage: [0, 0.2]
    def __init__(self, mag):
        minval = 0
        maxval = 40
        self.val = (mag / 30) * (maxval - minval) + minval

    def __call__(self, img):
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

    def __repr__(self):
        return '%s(magnitude=%.2f)' % \
               (self.__class__.__name__, self.val)


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


class Identity:
    def __init__(self, mag):
        pass

    def __call__(self, img):
        return img

    def __repr__(self):
        return '%s' % \
               (self.__class__.__name__,)


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
        TranslateX,
        TranslateY,
    ]

    return l


class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


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
            img_transformed = op(mag=mag)(img)
            img_transformed.save("transform_test/transformed_{}_{}.png".format(str(op), mag))

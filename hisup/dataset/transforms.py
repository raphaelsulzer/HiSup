import cv2
import torch
import torchvision
import numpy as np

from torchvision.transforms import functional as F
from skimage.transform import resize


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, ann=None, points=None):
        if ann is None:
            for t in self.transforms:
                image = t(image, points)
            return image
        for t in self.transforms:
            image, ann = t(image, ann, points)
        return image, ann


class Resize(object):
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, ann, points=None):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0

        sx = self.ann_width / ann['width']
        sy = self.ann_height / ann['height']
        ann['junc_ori'] = ann['junctions'].copy()
        ann['junctions'][:, 0] = np.clip(ann['junctions'][:, 0] * sx, 0, self.ann_width - 1e-4)
        ann['junctions'][:, 1] = np.clip(ann['junctions'][:, 1] * sy, 0, self.ann_height - 1e-4)
        ann['width'] = self.ann_width
        ann['height'] = self.ann_height
        ann['mask_ori'] = ann['mask'].copy()
        ann['mask'] = cv2.resize(ann['mask'].astype(np.uint8), (int(self.ann_width), int(self.ann_height)))

        return image, ann


class ResizeImage(object):
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image, ann=None, points=None):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        if ann is None:
            if points is None:
                return image
            else:
                return image, points

        if points is None:
            return image, ann
        else:
            return image, ann, points

class ToTensor(object):
    def __call__(self, image, anns=None, points=None):
        if anns is None:
            if points is None:
                return F.to_tensor(image)
            else:
                return F.to_tensor(image), F.to_tensor(points)

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)

        if points is None:
            return F.to_tensor(image), anns
        else:
            return F.to_tensor(image), anns, F.to_tensor(points)

class Normalize(object):
    def __init__(self, mean, std, to_255=False):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image, anns=None, points=None):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        if anns is None:
            if points is None:
                return image
            else:
                return image, points
        if points is None:
            return image, anns
        else:
            return image, anns, points

class Color_jitter(object):
    def __init__(self):
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=.5, hue=.1)
    def __call__(self, image, anns=None):
        image = self.jitter(image)
        if anns is None:
            return image
        return image, anns
import cv2
import torch
import torchvision
import numpy as np

from torchvision.transforms import functional as F
from skimage.transform import resize


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, points=None, ann=None):
        if ann is None:
            for t in self.transforms:
                image, points = t(image, points)
            return image, points
        for t in self.transforms:
            image, points, ann = t(image, points, ann)
        return image, points, ann


class ResamplePointCloud:

    def __init__(self, n_points=100000):
        self.n_points = n_points

    def __call__(self, image, points, ann=None):

        if points is None:
            if ann is None:
                return image, points
            else:
                return image, points, ann

        num_points = len(points)

        if num_points == self.n_points:
            return points
        elif num_points > self.n_points:
            indices = np.random.choice(num_points, self.n_points, replace=False)  # Subsample
        else:
            indices = np.random.choice(num_points, self.n_points, replace=True)  # Upsample

        if ann is None:
            return image, points[indices]
        else:
            return image, points[indices], ann



class Resize:
    def __init__(self, image_height, image_width, ann_height, ann_width):
        self.image_height = image_height
        self.image_width = image_width
        self.ann_height = ann_height
        self.ann_width = ann_width

    def __call__(self, image, points, ann):
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

        return image, points, ann


class ResizeImage:
    def __init__(self, image_height, image_width):
        self.image_height = image_height
        self.image_width = image_width

    def __call__(self, image, points, ann=None):
        image = resize(image, (self.image_height, self.image_width))
        image = np.array(image, dtype=np.float32) / 255.0
        if ann is None:
            return image, points
        return image, points, ann


class ToTensor:
    def __call__(self, image, points, anns=None):

        if points is not None:
            points = F.to_tensor(points).to(torch.float32)

        if anns is None:
            return F.to_tensor(image).to(torch.float32), points

        for key, val in anns.items():
            if isinstance(val, np.ndarray):
                anns[key] = torch.from_numpy(val)

        return F.to_tensor(image).to(torch.float32), points, anns


class Normalize:
    def __init__(self, mean, std, to_255=False):
        self.mean = mean
        self.std = std
        self.to_255 = to_255

    def __call__(self, image, points, anns=None):
        if self.to_255:
            image *= 255.0
        image = F.normalize(image, mean=self.mean, std=self.std)
        if anns is None:
            return image, points
        return image, points, anns

class Color_jitter:
    def __init__(self):
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=.5, hue=.1)
    def __call__(self, image, anns=None):
        image = self.jitter(image)
        if anns is None:
            return image
        return image, anns
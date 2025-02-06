import cv2
import random
import os.path as osp
import numpy as np
import os
import copclib as copc
import logging

from skimage import io
from pycocotools.coco import COCO
from shapely.geometry import Polygon
from torch.utils.data.dataloader import default_collate

from hisup.dataset.default_dataset import DefaultDataset
from hisup.utils.logger import make_logger

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class TrainDataset(DefaultDataset):

    pass
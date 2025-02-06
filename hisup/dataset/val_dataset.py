import os.path as osp
from PIL import Image
import logging

from hisup.dataset.default_dataset import DefaultDataset

class ValDataset(DefaultDataset):
    def __init__(self, root, ann_file, use_lidar, use_images, augment, transform, logging_level=logging.INFO):
        super().__init__(root, ann_file, use_lidar, use_images, augment, transform, logging_level)

        self.id_to_img_map = {k:v for k, v in enumerate(self.tiles)}

    

    def image(self, idx):
        img_id = self.id_to_img_map[idx]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        image = Image.open(osp.join(self.root,file_name)).convert('RGB')
        return image
    
    def get_img_info(self, index):
        img_id = self.id_to_img_map[index]
        img_data = self.coco.imgs[img_id]
        return img_data

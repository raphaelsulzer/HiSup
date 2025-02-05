from functools import partial
from .transforms import *
from . import train_dataset
from hisup.config.paths_catalog import DatasetCatalog
from . import val_dataset, test_dataset


def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ResamplePointCloud(cfg.DATASETS.PCD.N_POINTS),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
         ]
    )
    return transforms


def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
        [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                cfg.DATASETS.IMAGE.WIDTH,
                cfg.DATASETS.TARGET.HEIGHT,
                cfg.DATASETS.TARGET.WIDTH),
         ResamplePointCloud(cfg.DATASETS.PCD.N_POINTS),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255),
         ])
    args['rotate_f'] = cfg.DATASETS.ROTATE_F
    args['use_lidar'] = cfg.USE_LIDAR
    args['use_images'] = cfg.USE_IMAGES
    dataset = factory(**args)

    collate_fn = partial(train_dataset.collate_fn, use_lidar= cfg.USE_LIDAR, use_images=cfg.USE_IMAGES)

    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          num_workers=cfg.DATALOADER.NUM_WORKERS)
    return dataset


def build_train_dataset_multi(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(train_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
        [Resize(cfg.DATASETS.IMAGE.HEIGHT,
                cfg.DATASETS.IMAGE.WIDTH,
                cfg.DATASETS.TARGET.HEIGHT,
                cfg.DATASETS.TARGET.WIDTH),
         ToTensor(),
         Color_jitter(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255),
         ])
    args['rotate_f'] = cfg.DATASETS.ROTATE_F
    dataset = factory(**args)
    return dataset


def build_val_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ResamplePointCloud(cfg.DATASETS.PCD.N_POINTS),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
         ]
    )

    name = cfg.DATASETS.VAL[0]
    dargs = DatasetCatalog.get(name)
    factory = getattr(val_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = transforms
    dataset = factory(**args)
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=dataset.collate_fn,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return dataset, dargs['args']['ann_file']

def build_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ResamplePointCloud(cfg.DATASETS.PCD.N_POINTS),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                   cfg.DATASETS.IMAGE.PIXEL_STD,
                   cfg.DATASETS.IMAGE.TO_255)
         ]
    )

    name = cfg.DATASETS.TEST[0]
    dargs = DatasetCatalog.get(name)
    factory = getattr(test_dataset, dargs['factory'])
    args = dargs['args']
    args['transform'] = transforms
    dataset = factory(**args)
    dataset = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=dataset.collate_fn,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )
    return dataset, dargs['args']['ann_file']

from functools import partial
from .transforms import *
from . import default_dataset, train_dataset
from hisup.config.paths_catalog import DatasetCatalog
from . import val_dataset, test_dataset


def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         # ResamplePointCloud(cfg.DATASETS.PCD.N_POINTS),
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
    if cfg.MODEL.USE_IMAGES:
       transforms = Compose(
            [ResizeImageAndAnnotation(cfg.DATASETS.IMAGE.HEIGHT,
                                      cfg.DATASETS.IMAGE.WIDTH,
                                      cfg.DATASETS.TARGET.HEIGHT,
                                      cfg.DATASETS.TARGET.WIDTH),
             ToTensor(),
             Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                       cfg.DATASETS.IMAGE.PIXEL_STD,
                       cfg.DATASETS.IMAGE.TO_255),
             ])
    else:
       transforms = Compose(
           [ResizeImageAndAnnotation(cfg.DATASETS.IMAGE.HEIGHT,
                                     cfg.DATASETS.IMAGE.WIDTH,
                                     cfg.DATASETS.TARGET.HEIGHT,
                                     cfg.DATASETS.TARGET.WIDTH),
            ToTensor()]
       )


    args = dargs['args']
    args['transform'] = transforms
    args['augment'] = cfg.DATASETS.ROTATE_F
    args['use_lidar'] = cfg.MODEL.USE_LIDAR
    args['use_images'] = cfg.MODEL.USE_IMAGES
    dataset = factory(**args)

    collate_fn = partial(default_dataset.collate_fn, use_lidar= cfg.MODEL.USE_LIDAR, use_images=cfg.MODEL.USE_IMAGES)

    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=collate_fn,
                                          shuffle=True,
                                          num_workers=cfg.DATALOADER.NUM_WORKERS)
    return dataset



def build_val_dataset(cfg):

    if cfg.MODEL.USE_IMAGES:
        transforms = Compose(
            [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                         cfg.DATASETS.IMAGE.WIDTH),
             ToTensor(),
             Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                       cfg.DATASETS.IMAGE.PIXEL_STD,
                       cfg.DATASETS.IMAGE.TO_255)
             ]
        )
    else:
        transforms = Compose(
            [ToTensor()]
        )

    name = cfg.DATASETS.VAL[0]
    dargs = DatasetCatalog.get(name)
    factory = getattr(val_dataset, dargs['factory'])

    args = dargs['args']
    args['transform'] = transforms
    args['augment'] = False
    args['use_lidar'] = cfg.MODEL.USE_LIDAR
    args['use_images'] = cfg.MODEL.USE_IMAGES

    collate_fn = partial(default_dataset.collate_fn, use_lidar= cfg.MODEL.USE_LIDAR, use_images=cfg.MODEL.USE_IMAGES)

    dataset = factory(**args)
    dataset = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        collate_fn=collate_fn,
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

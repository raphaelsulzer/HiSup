import os.path as osp

class DatasetCatalog(object):

    # DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
    #             '..','..','data'))
    DATA_DIR = "./data"
    
    DATASETS = {
        'crowdai_train_small': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation-small.json'
        },
        'crowdai_test_small': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation-small.json'
        },
        'crowdai_train': {
            'img_dir': 'crowdai/train/images',
            'ann_file': 'crowdai/train/annotation.json'
        },
        'crowdai_test': {
            'img_dir': 'crowdai/val/images',
            'ann_file': 'crowdai/val/annotation.json'
        },
        'inria_train': {
            'img_dir': 'inria/train/images',
            'ann_file': 'inria/train/annotation.json',
        },
        'inria_test': {
            'img_dir': 'coco-Aerial/val/images',
            'ann_file': 'coco-Aerial/val/annotation.json',
        },
        'lidarpoly_train': {
            'img_dir': 'lidarpoly/train/images',
            'ann_file': 'lidarpoly/annotations_train.json',
        },
        'lidarpoly_val': {
            'img_dir': 'lidarpoly/val/images',
            'ann_file': 'lidarpoly/annotations_val.json',
        },
        'lidarpoly_test': {
            'img_dir': 'lidarpoly/test/images',
            'ann_file': 'lidarpoly/annotations_test.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'val' in name:
            return dict(factory="ValDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()

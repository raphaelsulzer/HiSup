import os
import logging
import torch
import torch.nn as nn
from pointpillars.model import PillarLayer, PillarEncoder, Backbone, Neck

from .bn_helper import BatchNorm2d_class
from hisup.backbones.multi_task_head import MultitaskHead


class PointCloudEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.logger = logging.getLogger("HiSup")
        
        res = cfg.DATASETS.IMAGE.HEIGHT
        voxel_size=(cfg.MODEL.POINT_ENCODER.voxel_size,cfg.MODEL.POINT_ENCODER.voxel_size,res)
        point_cloud_range=[0,0,0,res,res,res]
        max_voxels=(cfg.MODEL.POINT_ENCODER.max_voxels,cfg.MODEL.POINT_ENCODER.max_voxels) # (train,test)
        max_num_points = cfg.MODEL.POINT_ENCODER.max_num_points

        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=8,
                                            out_channel=64)

        layer_strides = [1 if cfg.MODEL.POINT_ENCODER.voxel_size == 4 else 2, 2, 2]
        self.pillar_backbone = Backbone(in_channel=64,
                                 out_channels=[64, 128, 256],
                                 layer_nums=[3, 5, 5],
                                        layer_strides=layer_strides)

        self.pillar_neck = Neck(in_channels=[64, 128, 256],
                         upsample_strides=[1, 2, 4],
                         out_channels=[128, 128, 128])

        # this is not really the head from PointPillars, I just give it that name because the parameters should be counted as backbone params
        head_size = cfg.MODEL.HEAD_SIZE
        num_class = sum(sum(head_size, []))
        self.head = MultitaskHead(input_channels=cfg.MODEL.OUT_FEATURE_CHANNELS,num_class=num_class,head_size=head_size)
        
        
    
    
    def init_weights(self, pretrained=''):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            model_dict = self.state_dict()              
            pretrained_dict = {k: v for k, v in pretrained_dict['model'].items()
                               if k in model_dict.keys()}
            if not pretrained_dict:
                self.logger.warning("Did not load any weights from LiDAR backbone.")
            # for k, _ in pretrained_dict.items():
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
    
    def forward(self, batched_pts):
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # print(f"Average number of pillars per sample: {pillars.shape[0]/len(batched_pts)}")
        # print(f"Average number of points per pillar: {npoints_per_pillar.cpu().numpy().mean()}")

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        pillar_features = self.pillar_backbone(pillar_features)

        pillar_features = self.pillar_neck(pillar_features)

        return pillar_features
        
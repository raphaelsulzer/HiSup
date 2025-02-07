import torch.nn as nn

# had to "pip install ." this (i.e. not develop install it, to be able to access it here)
# this however means that changes will not take effect here, and I have to reinstall every time
from pointpillars.model import PillarLayer, PillarEncoder

class LiDAR_Encoder(nn.Module):

    def __init__(self,
                 voxel_size=[1, 1, 4],
                 point_cloud_range=[0, -39.68, -3, 69.12, 39.68, 1],
                 max_num_points=16,
                 max_voxels=(16000, 16000)):
        super().__init__()

        self.pillar_layer = PillarLayer(voxel_size=voxel_size,
                                        point_cloud_range=point_cloud_range,
                                        max_num_points=max_num_points,
                                        max_voxels=max_voxels)

        self.pillar_encoder = PillarEncoder(voxel_size=voxel_size,
                                            point_cloud_range=point_cloud_range,
                                            in_channel=8,
                                            out_channel=64)

    def forward(self, batched_pts):
        # batched_pts: list[tensor] -> pillars: (p1 + p2 + ... + pb, num_points, c),
        #                              coors_batch: (p1 + p2 + ... + pb, 1 + 3),
        #                              num_points_per_pillar: (p1 + p2 + ... + pb, ), (b: batch size)
        pillars, coors_batch, npoints_per_pillar = self.pillar_layer(batched_pts)

        # pillars: (p1 + p2 + ... + pb, num_points, c), c = 4
        # coors_batch: (p1 + p2 + ... + pb, 1 + 3)
        # npoints_per_pillar: (p1 + p2 + ... + pb, )
        #                     -> pillar_features: (bs, out_channel, y_l, x_l)
        pillar_features = self.pillar_encoder(pillars, coors_batch, npoints_per_pillar)

        return pillar_features
from hisup.detector_default import *
from hisup.backbones.multi_task_head import MultitaskHead

from pointpillars.model import PillarLayer, PillarEncoder, Backbone, Neck

class LiDARBuildingDetector(BuildingDetector):
    def __init__(self, cfg):
        super().__init__(cfg)

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
        self.pillar_head = MultitaskHead(input_channels=cfg.MODEL.OUT_FEATURE_CHANNELS,num_class=num_class,head_size=head_size)

    def forward(self, images, points, annotations=None):
        if self.training:
            return self.forward_train(images, points, annotations=annotations)
        else:
            return self.forward_test(images, points)

    def forward_test(self, images, points):

        features, outputs = self.forward_points(points)

        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        joff_pred = outputs[:, :].sigmoid() - 0.5

        jloc_convex_pred = jloc_pred.softmax(1)[:, 2:3]
        jloc_concave_pred = jloc_pred.softmax(1)[:, 1:2]
        remask_pred = remask_pred.softmax(1)[:, 1:]

        scale_y = self.origin_height / self.pred_height
        scale_x = self.origin_width / self.pred_width

        batch_polygons = []
        batch_masks = []
        batch_scores = []
        batch_juncs = []

        for b in range(remask_pred.size(0)):
            mask_pred_per_im = cv2.resize(remask_pred[b][0].cpu().numpy(), (self.origin_width, self.origin_height))
            juncs_pred = get_pred_junctions(jloc_concave_pred[b], jloc_convex_pred[b], joff_pred[b])
            juncs_pred[:, 0] *= scale_x
            juncs_pred[:, 1] *= scale_y

            polys, scores = [], []
            props = regionprops(label(mask_pred_per_im > 0.5))
            for prop in props:
                poly, juncs_sa, edges_sa, score, juncs_index = generate_polygon(prop, mask_pred_per_im, \
                                                                                juncs_pred, 0, self.test_inria)
                if juncs_sa.shape[0] == 0:
                    continue

                polys.append(poly)
                scores.append(score)
            batch_scores.append(scores)
            batch_polygons.append(polys)

            batch_masks.append(mask_pred_per_im)
            batch_juncs.append(juncs_pred)

        extra_info = {}
        output = {
            'polys_pred': batch_polygons,
            'mask_pred': batch_masks,
            'scores': batch_scores,
            'juncs_pred': batch_juncs
        }
        return output, extra_info


    def forward_points(self, batched_pts):
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

        pillar_outputs = self.pillar_head(pillar_features)

        return pillar_features, pillar_outputs

    def forward_train(self, images, points, annotations=None):
        self.train_step += 1

        targets, metas = self.encoder(annotations)

        features, outputs = self.forward_points(points)

        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm': 0.0,
            'loss_remask': 0.0
        }

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature, mask_feature)
        jloc_att_feature = self.a2j_att(afm_feature, jloc_feature)

        mask_pred = self.mask_predictor(mask_feature + mask_att_feature)
        jloc_pred = self.jloc_predictor(jloc_feature + jloc_att_feature)
        afm_pred = self.afm_predictor(afm_feature)

        afm_conv = self.refuse_conv(afm_pred)
        remask_pred = self.final_conv(torch.cat((features, afm_conv), dim=1))

        if targets is not None:
            loss_dict['loss_jloc'] += F.cross_entropy(jloc_pred, targets['jloc'].squeeze(dim=1))
            loss_dict['loss_joff'] += sigmoid_l1_loss(outputs[:, :], targets['joff'], -0.5, targets['jloc'])
            loss_dict['loss_mask'] += F.cross_entropy(mask_pred, targets['mask'].squeeze(dim=1).long())
            loss_dict['loss_afm'] += F.l1_loss(afm_pred, targets['afmap'])
            loss_dict['loss_remask'] += F.cross_entropy(remask_pred, targets['mask'].squeeze(dim=1).long())
        extra_info = {}

        return loss_dict, extra_info
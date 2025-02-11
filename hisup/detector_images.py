from hisup.detector_default import *
from hisup.utils.polygon import generate_polygon
from hisup.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops

class ImageBuildingDetector(nn.Module):
    def __init__(self, cfg):
        super(BuildingDetector, self).__init__()
        self.backbone = build_backbone(cfg)
        self.backbone_name = cfg.MODEL.NAME
        self.test_inria = 'inria' in cfg.DATASETS.TEST[0]

        self.encoder = Encoder(cfg)

        self.pred_height = cfg.DATASETS.TARGET.HEIGHT
        self.pred_width = cfg.DATASETS.TARGET.WIDTH
        self.origin_height = cfg.DATASETS.ORIGIN.HEIGHT
        self.origin_width = cfg.DATASETS.ORIGIN.WIDTH

        dim_in = cfg.MODEL.OUT_FEATURE_CHANNELS
        self.mask_head = self._make_conv(dim_in, dim_in, dim_in)
        self.jloc_head = self._make_conv(dim_in, dim_in, dim_in)
        self.afm_head = self._make_conv(dim_in, dim_in, dim_in)

        self.a2m_att = ECA(dim_in)
        self.a2j_att = ECA(dim_in)

        self.mask_predictor = self._make_predictor(dim_in, 2)
        self.jloc_predictor = self._make_predictor(dim_in, 3)
        self.afm_predictor = self._make_predictor(dim_in, 2)

        self.refuse_conv = self._make_conv(2, dim_in//2, dim_in)
        self.final_conv = self._make_conv(dim_in*2, dim_in, 2)

        self.train_step = 0
        
    def forward(self, images, annotations = None):
        if self.training:
            return self.forward_train(images, annotations=annotations)
        else:
            return self.forward_test(images)

    def forward_test(self, images):

        outputs, features = self.backbone(images)

        mask_feature = self.mask_head(features)
        jloc_feature = self.jloc_head(features)
        afm_feature = self.afm_head(features)

        mask_att_feature = self.a2m_att(afm_feature, mask_feature)
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
            juncs_pred[:,0] *= scale_x
            juncs_pred[:,1] *= scale_y

            if not self.test_inria:
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


    def forward_train(self, images, annotations = None):
        self.train_step += 1

        targets, metas = self.encoder(annotations)
        outputs, features = self.backbone(images)

        loss_dict = {
            'loss_jloc': 0.0,
            'loss_joff': 0.0,
            'loss_mask': 0.0,
            'loss_afm' : 0.0,
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
    
    def _make_conv(self, dim_in, dim_hid, dim_out):
        layer = nn.Sequential(
            nn.Conv2d(dim_in, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_hid, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_hid),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_hid, dim_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True),
        )
        return layer

    def _make_predictor(self, dim_in, dim_out):
        m = int(dim_in / 4)
        layer = nn.Sequential(
                    nn.Conv2d(dim_in, m, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(m, dim_out, kernel_size=1),
                )
        return layer


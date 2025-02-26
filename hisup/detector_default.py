import cv2
import torch
import torch.nn.functional as F

from math import log
from torch import nn
from hisup.encoder import Encoder
from hisup.utils.polygon import generate_polygon
from hisup.utils.polygon import get_pred_junctions
from skimage.measure import label, regionprops


def cross_entropy_loss_for_junction(logits, positive):
    nlogp = -F.log_softmax(logits, dim=1)

    loss = (positive * nlogp[:, None, 1] + (1 - positive) * nlogp[:, None, 0])

    return loss.mean()

def sigmoid_l1_loss(logits, targets, offset = 0.0, mask=None):
    logp = torch.sigmoid(logits) + offset
    loss = torch.abs(logp-targets)

    if mask is not None:
        t = ((mask == 1) | (mask == 2)).float()
        w = t.mean(3, True).mean(2,True)
        w[w==0] = 1
        loss = loss*(t/w)

    return loss.mean()

class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        C = channel

        t = int(abs((log(C, 2) + b) / gamma))
        k = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
        self.sigmoid = nn.Sigmoid()

        self.out_conv = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        y = self.avg_pool(x1 + x2)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1 ,-2).unsqueeze(-1)
        y = self.sigmoid(y)

        out = self.out_conv(x2 * y.expand_as(x2))
        return out


class BuildingDetector(nn.Module):
    def __init__(self, cfg):
        super(BuildingDetector, self).__init__()

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


    def jloc_vis(self, tensor):
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        # Define color map based on the range of values {0, 1, 2}
        cmap = mcolors.ListedColormap([(0.5, 0.5, 0.5, 0.5), 'green', 'red'])
        bounds = [0, 1, 2, 3]  # Define the range of values
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # Plotting the tensor values as an image
        plt.imshow(tensor.numpy(), cmap=cmap, norm=norm)

        # Add legend manually and place it outside the axis
        legend_labels = ['outside', 'concave', 'convex']
        colors = ['grey', 'green', 'red']
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in
                   colors]
        plt.legend(handles, legend_labels, title="Value", loc='upper left', bbox_to_anchor=(1, 1))

        # Title and layout
        plt.title("jloc Tensor")
        plt.axis('off')  # Optionally turn off the axis
        plt.tight_layout()

        # Show the plot
        plt.show()


def get_pretrained_model(cfg, file, device):

    print(f"Loading pretrained model from {file}")

    model = BuildingDetector(cfg, test=True)
    state_dict = torch.load(file, map_location=device)
    # state_dict = state_dict["model"]
    state_dict = {k[7:]:v for k,v in state_dict['model'].items() if k[0:7] == 'module.'}
    model.load_state_dict(state_dict)
    model = model.eval()
    return model


def get_pretrained_model_from_url(cfg, dataset, device):
    PRETRAINED = {
        'crowdai': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/crowdai_hrnet48_e100.pth',
        'inria': 'https://github.com/XJKunnn/pretrained_model/releases/download/pretrained_model/inria_hrnet48_e5.pth',
    }

    model = BuildingDetector(cfg, test=True)
    url = PRETRAINED[dataset]
    state_dict = torch.hub.load_state_dict_from_url(url, map_location=device, progress=True)
    state_dict = {k[7:]:v for k,v in state_dict['model'].items() if k[0:7] == 'module.'}
    model.load_state_dict(state_dict)
    model = model.eval()

    return model


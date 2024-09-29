import torch.nn as nn
import torch
import torch.nn.functional as F
from .guided_filter import GuidedFilter
from .agent_swin import AgentAttention

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x

def _upsample_like_64(src):
    src = F.interpolate(src, size=(64, 64), mode='bilinear')
    return src
def _upsample_like_128(src):
    src = F.interpolate(src, size=(128, 128), mode='bilinear')
    return src
def _upsample_like_256(src):
    src = F.interpolate(src, size=(256, 256), mode='bilinear')
    return src


class Loop_Finer(nn.Module):
    def __init__(self, mid_ch):
        super(Loop_Finer, self).__init__()
        self.gfp = GuidedFilter()
        self.finer0 = BasicConv2d(264, mid_ch, kernel_size=3, stride=1, padding=1)
        self.finer1 = BasicConv2d(264, mid_ch, kernel_size=3, stride=1, padding=1)
        self.finer2 = BasicConv2d(mid_ch*2, mid_ch, kernel_size=1, stride=1, padding=1)
        self.finer_atten = AgentAttention(dim=64, num_heads=2)
        self.finer4 = BasicConv2d(mid_ch, 1, kernel_size=1, stride=1, padding=0)  # fini_ch
    def forward(self, pred, image_em, depth_em):
        pred = _upsample_like_64(pred)
        reversed_pred = 1 - pred
        image_cut = torch.chunk(image_em, 8, dim=1)
        image_cat = torch.cat((image_cut[0], reversed_pred, image_cut[1], reversed_pred, image_cut[2], reversed_pred, image_cut[3], reversed_pred
                               , image_cut[4], reversed_pred, image_cut[5], reversed_pred, image_cut[6], reversed_pred, image_cut[7], reversed_pred), 1)

        depth_cut = torch.chunk(depth_em, 8, dim=1)
        depth_cat = torch.cat((depth_cut[0], reversed_pred, depth_cut[1], reversed_pred, depth_cut[2], reversed_pred, depth_cut[3], reversed_pred
                               , depth_cut[4], reversed_pred, depth_cut[5], reversed_pred, depth_cut[6], reversed_pred, depth_cut[7], reversed_pred), 1)

        image_fliter = self.gfp(image_cat, depth_cat)
        image_fliter = image_fliter + image_cat

        i_f0 = self.finer0(image_fliter)
        i_f0 = _upsample_like_64(i_f0)

        d_f1 = self.finer1(depth_cat)
        d_f1 = _upsample_like_64(d_f1)
        tmp = d_f1
        d_f1 = self.finer_atten(d_f1)
        d_f1 = d_f1 + tmp

        i_f2 = self.finer2(torch.cat((i_f0, d_f1), 1))
        i_f2 = _upsample_like_128(i_f2)

        d_f3 = self.finer_atten(i_f2)
        d_f3 = _upsample_like_128(d_f3)
        d_f4 = self.finer4(d_f3)
        d_f4 = _upsample_like_256(d_f4)

        return d_f4
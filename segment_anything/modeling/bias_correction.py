import torch.nn as nn
from segment_anything.modeling.PUA_res_plus import PUAModule
import torch.nn.functional as F
import torch

class CrossAttention(nn.Module):
    def __init__(self, in_channels):
        super(CrossAttention, self).__init__()

        # Query, Key, and Value projections for both input vectors
        self.query_v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_v1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.query_v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key_v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value_v2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, v1, v2):
        # Project vectors to Query, Key, and Value
        query_v1 = self.query_v1(v1)
        key_v2 = self.key_v2(v2)
        value_v2 = self.value_v2(v2)

        # Compute attention scores
        scores = torch.matmul(query_v1.view(query_v1.size(0), -1, query_v1.size(-1)),
                              key_v2.view(key_v2.size(0), -1, key_v2.size(-1)).transpose(1, 2))
        attention = F.softmax(scores, dim=-1)

        # Apply attention to values
        output_v1 = torch.matmul(attention, value_v2.view(value_v2.size(0), -1, value_v2.size(-1)))
        output_v1 = output_v1.view(v1.size())

        return output_v1

def _upsample_like_64(src):
    src = F.interpolate(src, size=(64,64), mode='bilinear')
    return src

class bias_correction(nn.Module):
    def __init__(self, out_channels):
        super(bias_correction, self).__init__()
        self.PUA = PUAModule(out_channels)
        self.conv = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, stride=1, padding=1)
    def forward(self, embedding):
        embedding_64 = _upsample_like_64(embedding)
        pua_em = self.PUA(embedding)
        embedding = self.conv(embedding)
        pua_em = self.conv(pua_em)
        pua_em = _upsample_like_64(pua_em)
        embedding = _upsample_like_64(embedding)
        out_em = pua_em + embedding
        return out_em, embedding_64


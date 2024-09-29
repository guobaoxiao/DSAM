import torch
import torch.nn as nn
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, need_relu=True,
                 bn=nn.BatchNorm2d):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = bn(out_channels)
        self.relu = nn.ReLU()
        self.need_relu = need_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.need_relu:
            x = self.relu(x)
        return x
class ME(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ME, self).__init__()

        self.depthwise_conv_reduce_channels_1 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=1,
                                                               stride=1, padding=0, groups=in_channels // 16)
        self.depthwise_conv_reduce_channels_2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                                               stride=1, padding=0, groups=in_channels // 16)
        self.relu = nn.ReLU(True)
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channels, out_channels, 1),
            BasicConv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channels, out_channels, 3, padding=3, dilation=3)
        )

    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:

                if len(param.shape) == 1:
                    param_unsqueeze = param.unsqueeze(0)
                    nn.init.xavier_uniform_(param_unsqueeze)
                    param.data.copy_(param_unsqueeze.squeeze(0))
                else:
                    nn.init.xavier_uniform_(param)

            elif 'bias' in name:
                # print("bias:" + name)
                # The bias term is initialized
                nn.init.zeros_(param)
    def forward(self,dense_embeddings_box, high_frequency, sparse_embeddings_box):  # high_frequency, 这里删了一个这个

        # dense_cat = torch.cat([dense_embeddings_boundary, dense_embeddings_box], dim=1)
        dense_cat_tmp = self.depthwise_conv_reduce_channels_1(dense_embeddings_box)
        dense_em = self.branch1(dense_embeddings_box)
        dense_em =dense_em + dense_cat_tmp

        dense_embeddings = torch.cat([dense_em, high_frequency], dim=1)  # 这里注销了一个这个

        dense_embeddings = self.depthwise_conv_reduce_channels_2(dense_embeddings)
        sparse_embeddings = sparse_embeddings_box
        return dense_embeddings, sparse_embeddings

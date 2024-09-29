import torch.nn as nn
import torch

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
# merge_dense_sparse
class MDS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MDS, self).__init__()
        '''8 :sparse_embedding:tensor(bs, 2, 256) -> tensor(bs, 1, 2, 256) 
             :dense_embedding:tensor(bs, 256, 64, 64) 
             8 = 64*64/2/256'''
        self.sparse_in_channels = 1
        self.dense_in_channels = in_channels*8
        self.real_in_channels = in_channels*8+self.sparse_in_channels
        self.real_out_channels = out_channels*8+self.sparse_in_channels
        self.relu = nn.ReLU(True)
        # self.branch0 = BasicConv2d(self.real_in_channels, self.real_out_channels, 1)
        self.branch1 = nn.Sequential(
            BasicConv2d(self.real_in_channels, self.real_out_channels, 1),
            BasicConv2d(self.real_out_channels, self.real_out_channels, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(self.real_out_channels, self.real_out_channels, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(self.real_out_channels, self.real_out_channels, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(self.real_in_channels, self.real_out_channels, 1),
            BasicConv2d(self.real_out_channels, self.real_out_channels, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(self.real_out_channels, self.real_out_channels, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(self.real_out_channels, self.real_out_channels, 3, padding=5, dilation=5)
        )
        # self.branch3 = nn.Sequential(
        #     BasicConv2d(in_channels, out_channels, 1),
        #     BasicConv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3)),
        #     BasicConv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0)),
        #     BasicConv2d(out_channels, out_channels, 3, padding=7, dilation=7)
        # )
        self.conv_cat = BasicConv2d(2 * self.real_out_channels, self.real_out_channels, 3, padding=1)
        # self.conv_res = BasicConv2d(in_channels, out_channels, 1)
        self.conv_spares_cat = BasicConv2d(2*self.sparse_in_channels, self.sparse_in_channels, 3, padding=1)
        self.conv_dense_cat = BasicConv2d(2*self.dense_in_channels, self.dense_in_channels, 3, padding=1)
    def initialize_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                # Initialize the weight parameters
                # One-dimensional vectors cannot be initialized, so we first convert them to two dimensions,
                # initialize them, and then convert them to one dimension
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

    def forward(self, sparse_embeddings, dense_embeddings):
        sparse_embeddings = sparse_embeddings.unsqueeze(1)
        bs_d, c_d, h_d, w_d = dense_embeddings.size()
        bs_s, c_s, h_s, w_s = sparse_embeddings.size()
        n = h_d * w_d // (h_s * w_s)
        dense_embeddings = dense_embeddings.view(bs_d, n*c_d, h_s, w_s)
        tmp_sparse = sparse_embeddings
        tmp_dense = dense_embeddings
        sd_cat = torch.cat([sparse_embeddings, dense_embeddings], dim=1)

        # x0 = self.branch0(sd_cat)
        x1 = self.branch1(sd_cat)
        x_cat = x1
        # x_cat = self.conv_cat(torch.cat((x0, x1), 1))

        sparse_embeddings_up = x_cat[:, :self.sparse_in_channels, :, :]
        sparse_embeddings = self.relu(sparse_embeddings * sparse_embeddings_up)
        sparse_cat = torch.cat([sparse_embeddings, tmp_sparse], dim=1)
        sparse_embeddings = self.conv_spares_cat(sparse_cat)
        # sparse_embeddings = sparse_embeddings.view(32, 1, -1, 256)
        sparse_embeddings = sparse_embeddings.squeeze(dim=1)

        dense_embeddings_down = x_cat[:, self.sparse_in_channels:, :, :]
        dense_embeddings = self.relu(dense_embeddings * dense_embeddings_down)
        dense_embeddings = self.conv_dense_cat(torch.cat([dense_embeddings, tmp_dense], dim=1))
        dense_embeddings = dense_embeddings.view(bs_d, c_d, h_d, w_d)
        # dense_embeddings = dense_embeddings + tmp_dense
        return sparse_embeddings, dense_embeddings
from __future__ import print_function
import math
import copy
from typing import Optional, List
import torch
from torch import Tensor
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.models
from .netvlad import NetVLAD


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, not_mask):
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=not_mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # for q and k, scr + pos; for v, src
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        #Add and Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        #FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        #Add and Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # Norm
        src2 = self.norm1(src)
        # for q and k, scr + pos; for v, src
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # Add and Norm
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        # FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # Add
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        # copy 6 times
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # src is the image feature，shape=hxw,b,256
        output = src
        for layer in self.layers:
            # for each encoder layer inside, adding the position embedding.
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)
        return output


class NetWork(nn.Module):
    def __init__(self, output_dim=4096, emb_dims=512, num_clusters=64, layer_number=3, learned_position_encoding=False):
        super(NetWork, self).__init__()

        # self.feature_encoder = nn.Sequential(*list(torchvision.models.alexnet(pretrained=True).children())[:-2])
        self.feature_encoder = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).children())[:-2])
        # self.feature_encoder = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
        # self.feature_encoder = nn.Sequential(*list(torchvision.models.mobilenet_v2(pretrained=True).children())[:-1])

        self.learned_position_encoding = learned_position_encoding
        self.PositionEmbeddingSine = PositionEmbeddingSine(num_pos_feats=256, temperature=10000)
        self.PositionEmbeddingLearned = PositionEmbeddingLearned(num_pos_feats=256)

        encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1,
                                                activation="relu", normalize_before=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=layer_number, norm=None)

        self.net_vlad = NetVLAD(num_clusters=num_clusters, dim=emb_dims)

        self.hidden_weights = nn.Parameter(torch.randn(num_clusters * emb_dims, output_dim) * 1 / math.sqrt(emb_dims))
        # self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = torch.unsqueeze(x, dim=1)
        x = x.repeat(1, 3, 1, 1)
        x = self.feature_encoder(x)

        if self.learned_position_encoding:
            PositionEmbedding = self.PositionEmbeddingLearned(x)
        else:
            not_mask = torch.ones(x.shape[-2], x.shape[-1]).unsqueeze(0).cuda()
            PositionEmbedding = self.PositionEmbeddingSine(not_mask)

        b = x.shape[0]
        c = x.shape[1]
        h = x.shape[2]
        w = x.shape[3]
        x = x.flatten(2).permute(2, 0, 1)
        PositionEmbedding = PositionEmbedding.flatten(2).permute(2, 0, 1)

        x = self.transformer_encoder(src=x, pos=PositionEmbedding) # the format of input tensor is sequence_length * batch_number * feature_dim

        x = x.permute(1, 2, 0)
        x = x.reshape(b, c, h, w)

        x = self.net_vlad(x)

        # dimention reduction
        x = torch.matmul(x, self.hidden_weights)
        # x = self.bn(x)
        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    num_image = 12
    image_width = 400
    image_high = 300
    sim_data = Variable(torch.rand(batch_size, num_image, image_high, image_width))
    sim_data = sim_data.cuda()

    nnv = NetWork(output_dim=512, emb_dims=512, layer_number=3).cuda()
    nnv.train()
    out = nnv(sim_data)

    print('pnv', out.size())

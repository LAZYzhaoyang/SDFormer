"""
# Code of paper: "SDFormer: A Novel Transformer Neural Network for Structural Damage Identification by Segmenting The Strain Field Map".
# author: Zhaoyang Li
# Central South University, Changsha, China
# Lastest update: 2022/03/19
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from torch import einsum

import numpy as np

#============================Basic Block============================#
class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding_mode='same', relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        if padding_mode == 'same':
            padding = (kernel_size-1) // 2
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)
    
class ConvPool(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2):
        super(ConvPool, self).__init__()
        self.in_channel = in_channels
        self.out_channel = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.padding = 0
        self.pool = BasicConv(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=kernel_size, 
                              padding_mode='zero', relu=False, bn=False, bias=False)
        
    def forward(self, x):
        x = self.pool(x)
        return x

#============================Net Block============================#
class Spatial_Attention_Module(nn.Module):
    def __init__(self, channel):
        super(Spatial_Attention_Module, self).__init__()
        self.spatial_conv = BasicConv(in_channels=channel, out_channels=channel, kernel_size=1, relu=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        value = x
        attn = self.spatial_conv(x)
        scale = self.sigmoid(attn)
        out = value * scale
        
        return out

class Channel_Attention_Module(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(Channel_Attention_Module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y1 = self.avg_pool(x).view(b, c) # squeeze操作
        y1 = self.fc1(y1).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        y2 = self.max_pool(x).view(b, c)
        y2 = self.fc2(y2).view(b, c, 1, 1)
        y = y1 + y2
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return  out 
    
class Non_Local_Module(nn.Module):
    def __init__(self, channel):
        super(Non_Local_Module, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c, -1)
        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x # 残差连接
        return out
    
class Residual_Block(nn.Module):
    def __init__(self, channel):
        super(Residual_Block, self).__init__()
        self.convblock1 = BasicConv(in_channels=channel, out_channels=channel, bias=True)
        self.convblock2 = BasicConv(in_channels=channel,out_channels=channel, relu=False, bias=True)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = out + residual
        out = self.relu(out)
        
        return out
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, channels, scale_factor=2, bilinear=False):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)
            self.conv = BasicConv(in_channels=channels, out_channels=channels//scale_factor, bias=False, bn=False)
        else:
            self.up = nn.ConvTranspose2d(channels, channels // scale_factor, kernel_size=scale_factor, stride=scale_factor)
            
    def forward(self, x):

        out = self.up(x)
        if self.bilinear:
            out = self.conv(out)
        return out

class SegHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(SegHead, self).__init__()
        self.conv = BasicConv(in_channels=in_channels, out_channels=n_classes,relu=False, bn=False, bias=False)
        self.activation = nn.Softmax2d()

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)

        return out


#============================Networks Block============================#
# backbone

class resnet_down(nn.Module):
    def __init__(self, in_channel, out_channel, down_factor=2):
        super(resnet_down, self).__init__()
        self.resnet = Residual_Block(channel=out_channel)
        self.down = ConvPool(in_channels=in_channel, out_channels=out_channel, kernel_size=down_factor)

    def forward(self, x):
        out = self.down(x)
        out = self.resnet(out)

        return out


class resnet_up(nn.Module):
    def __init__(self, in_channel, up_factor=2, bilinear=True):
        super(resnet_up, self).__init__()
        self.up = Up(channels=in_channel, scale_factor=up_factor, bilinear=bilinear)
        self.resnet = Residual_Block(channel= in_channel//up_factor)

    def forward(self, x):
        out = self.up(x)
        out = self.resnet(out)
        return out

class CBAM_Block(nn.Module):
    def __init__(self, in_channel, is_parallel=False, mode='channel_first'):
        super(CBAM_Block, self).__init__()
        self.spatial_att = Spatial_Attention_Module(channel=in_channel)
        self.channel_att = Channel_Attention_Module(in_channel=in_channel)
        self.is_parallel = is_parallel
        self.mode = mode

    def forward(self, x):
        if self.is_parallel:
            spa_att = self.spatial_att(x)
            channel_att = self.channel_att(x)
            out = torch.cat([spa_att, channel_att], dim=1)
        else:
            if self.mode == 'channel_first':
                att = self.channel_att(x)
                out = self.spatial_att(att)
            else:
                att = self.spatial_att(x)
                out = self.channel_att(att)
        return out

#============================Visual Transformer Block============================#

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout)))
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x


'''
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
'''
class VisionTransformerBlock(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        if type(image_size) == int:
            h, w = image_size, image_size
        else:
            h, w = image_size[0], image_size[1]
        
        if type(patch_size) == int:
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size[0], patch_size[1]
        
        assert h % ph == 0, 'Image dimensions must be divisible by the patch size.'
        assert w % pw == 0, 'Image dimensions must be divisible by the patch size.'
        
        num_patches = (h//ph) * (w//pw)
        patch_dim = channels * ph * pw
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = ph, p2 = pw),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img, mask = None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)


        return x

class SegMlpHead(nn.Module):
    def __init__(self, dim, n_class, image_size, patch_size):
        super().__init__()
        if type(image_size) == int:
            h, w = image_size, image_size
        else:
            h, w = image_size[0], image_size[1]
            
        if type(patch_size) == int:
            ph, pw = patch_size, patch_size
        else:
            ph, pw = patch_size[0], patch_size[1]
        
        assert h % ph == 0, 'Image dimensions must be divisible by the patch size.'
        assert w % pw == 0, 'Image dimensions must be divisible by the patch size.'
        
        patch_dim = ph * pw * n_class
        nh, nw = h//ph, w//pw
        self.segmlp = nn.Linear(dim, patch_dim)
        self.rearrange = Rearrange('b (nh nw) (p1 p2 c) -> b c (nh p1) (nw p2)', nh=nh, nw=nw, p1=ph, p2=pw, c=n_class)
    
    def forward(self, x):
        x = self.segmlp(x)
        x = x[:, 1:, :]
        out = self.rearrange(x)
        
        return out

#============================Swin Visual Transformer Block============================#
class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x,
                          shifts=(self.displacement[0], self.displacement[1]), 
                          dims=(1, 2))


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size[0] *window_size[1], window_size[0] *window_size[1])

    if upper_lower:
        mask[-displacement[0] * window_size[0]:, :-displacement[1] * window_size[1]] = float('-inf')
        mask[:-displacement[0] * window_size[0], -displacement[1] * window_size[1]:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size[0], h2=window_size[1])
        mask[:, -displacement[0]:, :, :-displacement[1]] = float('-inf')
        mask[:, :-displacement[0], :, -displacement[1]:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(int(window_size[0])) for y in range(int(window_size[1]))]))
    distances = indices[None, :, :] - indices[:, None, :]
    distances[:,:,0] = distances[:,:,0] + window_size[0] - 1
    distances[:,:,1] = distances[:,:,1] + window_size[1] - 1
    distances = distances.to(torch.long)
    return distances

class WindowAttention(nn.Module):
    def __init__(self,
                 dim,
                 heads, 
                 head_dim, 
                 shifted, 
                 window_size, 
                 relative_pos_embedding):
        super().__init__()
        inner_dim = head_dim * heads

        if isinstance(window_size, int):
            window_size = [window_size, window_size]
            window_size = np.array(window_size)
        elif isinstance(window_size, list) or isinstance(window_size, tuple):
            window_size = np.array(window_size)
        else:
            ValueError('window_size must be one of tuple, list or int')
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # change the window local by shifted the image
        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size)
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size[0] *window_size[1], window_size[0] *window_size[1]))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            x = self.cyclic_shift(x)
        # x : [b, n_h, n_w, dim]
        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # x : [b, n_h, n_w, inner_dim, 3]
        nw_h = n_h // self.window_size[0]
        nw_w = n_w // self.window_size[1]

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size[0], w_w=self.window_size[1]), qkv)
        # q, k, v : [b, head, windows_num(nw_h*nw_w), windows_size**2, head_dim]
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        # dots : [b, head, windows_num, windows_size**2, windows_size**2]
        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding
        
        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask
        
        attn = dots.softmax(dim=-1)
        # attn.shape = dots.shape
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        # out: [b, head, windows_num(nw_h*nw_w), windows_size**2, head_dim]
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size[0], w_w=self.window_size[1], nw_h=nw_h, nw_w=nw_w)
        # out: [b, n_h, n_w, inner_dim]
        out = self.to_out(out)
        # out: [b, n_h, n_w, out_dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out


class SwinBlock(nn.Module):
    def __init__(self,
                 dim, 
                 heads, 
                 head_dim, 
                 mlp_dim, 
                 shifted, 
                 window_size, 
                 relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        # x: [b, n_h, n_w, dim]
        x = self.attention_block(x)
        # x: [b, n_h, n_w, out_dim]
        x = self.mlp_block(x)
        # x: [b, n_h, n_w, out_dim]
        return x


class PatchMerging(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        # Unfold: input: [N, C, H, W], output: [N, C*kernel_size[0]*kernel_size[1], L]
        # L = ((h-2*padding-(kernel_size[0]-1)-1)/stride[0]+1) * 
        #       ((w-2*padding-(kernel_size[1]-1)-1)/stride[1]+1)
        # in here: L = (h*w)/(downscaling_factor**2)
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        #print(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        # view == resize, permute == transpose
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        # x [N, new_h, new_w, C*k1*k2]
        # looks like channel attention 
        x = self.linear(x)
        # output x : [N new_h, new_w, out_channels]
        return x


class UpSwinModule(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 up_scale_factor,
                 num_heads, 
                 head_dim, 
                 window_size,
                 relative_pos_embedding):
        super().__init__()
        # up_scale_factor: int
        #assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.layers = nn.ModuleList([])
        
        self.layers.append(nn.ModuleList([
            SwinBlock(dim=out_channel,
                      heads=num_heads, 
                      head_dim=head_dim, 
                      mlp_dim=out_channel * 4,
                      shifted=False, window_size=window_size, 
                      relative_pos_embedding=relative_pos_embedding),
            SwinBlock(dim=out_channel,
                      heads=num_heads, 
                      head_dim=head_dim, 
                      mlp_dim=out_channel * 4,
                      shifted=True, 
                      window_size=window_size, 
                      relative_pos_embedding=relative_pos_embedding),
        ]))
        self.add_channel = nn.Linear(in_channel, out_channel*up_scale_factor*up_scale_factor)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=up_scale_factor)
    
    def forward(self, x):
        # x: [N, in_C, H, W]
        x = x.permute(0, 2, 3, 1)
        # x: [N, H, W, in_C]
        x = self.add_channel(x)
        # x: [N, H, W, out_C*upscale^2]
        x = x.permute(0, 3, 1, 2)
        # x: [N, out_C*upscale^2, H, W]
        x = self.pixelshuffle(x)
        # x: [N, out_C, H*upscale, W*upscale]
        x = x.permute(0, 2, 3, 1)
        # x: [N, H*upscale, W*upscale, out_C]
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        # x: [N, H*upscale, W*upscale, out_C]
        x = x.permute(0, 3, 1, 2)
        # x: [N, out_C, H*upscale, W*upscale]

        return x


class StageModule(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_dimension, 
                 layers, 
                 downscaling_factor, 
                 num_heads, 
                 head_dim, 
                 window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels,
                                            out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension,
                          heads=num_heads, 
                          head_dim=head_dim, 
                          mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, 
                          relative_pos_embedding=relative_pos_embedding),
                SwinBlock(dim=hidden_dimension,
                          heads=num_heads, 
                          head_dim=head_dim, 
                          mlp_dim=hidden_dimension * 4,
                          shifted=True, 
                          window_size=window_size, 
                          relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x):
        # x: [b, c, h, w]
        x = self.patch_partition(x)
        # x: [b, new_h, new_w, hidden_dimension]
        # new_h = h // downscaling_factor
        # new_w = w // downscaling_factor
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)

            x = shifted_block(x)

        # x: [b, new_h, new_w, hidden_dimension]
        x = x.permute(0, 3, 1, 2)
        # x: [b, hidden_dimension, new_h, new_w]
        return x

class SwinTransformerEncoder(nn.Module):
    def __init__(self, *, 
                 hidden_dim, 
                 layers, 
                 heads, 
                 channels=3, 
                 head_dim=32, 
                 window_size=8,
                 downscaling_factors=(4, 2, 2, 2), 
                 relative_pos_embedding=True):
        super().__init__()
        if isinstance(layers, list) or isinstance(layers, tuple):
            assert len(layers)==len(downscaling_factors), 'len(layers) must equal len(downscaling_factors) !'
        elif isinstance(layers, int):
            nlayers = [layers for _ in range(len(downscaling_factors))]
            layers = nlayers
        else:
            TypeError('layers must be one of (tuple, list, int)')

        if isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            assert len(hidden_dim)==len(downscaling_factors), 'len(hidden_dim) must equal len(downscaling_factors) !'
        elif isinstance(hidden_dim, int):
            nhid = [hidden_dim*(2**i) for i in range(len(downscaling_factors))]
            hidden_dim = nhid
        else:
            TypeError('hidden_dim must be one of (tuple, list, int)')

        if isinstance(heads, list) or isinstance(heads, tuple):
            assert len(heads)==len(downscaling_factors), 'len(heads) must equal len(downscaling_factors) !'
        elif isinstance(heads, int):
            nheads = [heads for _ in range(len(downscaling_factors))]
            heads = nheads
        else:
            TypeError('heads must be one of (tuple, list, int)')
        
        if isinstance(window_size, list) or isinstance(window_size, tuple):
            assert len(window_size)==len(downscaling_factors), 'len(window_size) must equal len(downscaling_factors) !'
        elif isinstance(window_size, int):
            nws = [window_size for _ in range(len(downscaling_factors))]
            window_size = nws
        else:
            TypeError('window_size must be one of (tuple, list, int)')
        
        in_channel = channels
        self.stages = nn.ModuleList([])
        #print('encoder windowsize: ', window_size)

        for i in range(len(downscaling_factors)):
            out_channel = hidden_dim[i]
            self.stages.append(StageModule(in_channels=in_channel, hidden_dimension=out_channel, layers=layers[i],
                                           downscaling_factor=downscaling_factors[i], num_heads=heads[i], head_dim=head_dim,
                                           window_size=window_size[i], relative_pos_embedding=relative_pos_embedding))
            in_channel = out_channel

    def forward(self, x):
        out = []
        for stage in self.stages:
            x = stage(x)
            out.insert(0,x)
        return out

class SwinTransformerDecoder(nn.Module):
    def __init__(self, 
                 inchannels, 
                 hidden_dim, 
                 heads, 
                 head_dim=32, 
                 window_size=8, 
                 upscale_factors=(2, 2, 2, 4), 
                 relative_pos_embedding=True,
                 skip_connect=True):
        super().__init__()
        if isinstance(hidden_dim, list) or isinstance(hidden_dim, tuple):
            assert len(hidden_dim)==len(upscale_factors), 'len(hidden_dim) must equal len(downscaling_factors) !'
        elif isinstance(hidden_dim, int):
            nhid = [hidden_dim*(2**i) for i in range(len(upscale_factors))]
            hidden_dim = nhid
        else:
            TypeError('hidden_dim must be one of (tuple, list, int)')

        if isinstance(heads, list) or isinstance(heads, tuple):
            assert len(heads)==len(upscale_factors), 'len(heads) must equal len(downscaling_factors) !'
        elif isinstance(heads, int):
            nheads = [heads for _ in range(len(upscale_factors))]
            heads = nheads
        else:
            TypeError('layers must be one of (tuple, list, int)')

        if isinstance(window_size, list) or isinstance(window_size, tuple):
            assert len(window_size)==len(upscale_factors), 'len(window_size) must equal len(downscaling_factors) !'
        elif isinstance(window_size, int):
            nws = [window_size for _ in range(len(upscale_factors))]
            window_size = nws
        else:
            TypeError('window_size must be one of (tuple, list, int)')

        in_channel = inchannels

        self.decoders = nn.ModuleList([])
        #print('decoder windowsize: ', window_size)

        for i in range(len(upscale_factors)):
            out_channel = hidden_dim[i]
            self.decoders.append(UpSwinModule(in_channel=in_channel, 
                                              out_channel=out_channel,
                                              up_scale_factor=upscale_factors[i],
                                              num_heads=heads[i],
                                              head_dim=head_dim,
                                              window_size=window_size[i],
                                              relative_pos_embedding=relative_pos_embedding))
            in_channel = out_channel
            if skip_connect:
                in_channel = out_channel*2
        
        self.skip_connect = skip_connect

    def forward(self, x):
        if self.skip_connect:
            #print(x[0].shape)
            out = self.decoders[0](x[0])
            for i in range(len(self.decoders)-1):
                out = torch.cat((out, x[i+1]), dim=1)
                out = self.decoders[i+1](out)
        else:
            for i in range(len(self.decoders)):
                x = self.decoders[i](x)
            out = x
        return out

#============================Swin Visual Transformer============================#

class SwinTransformer(nn.Module):
    def __init__(self, *, hidden_dim, layers, heads, channels=3, num_classes=1000, head_dim=32, window_size=7,
                 downscaling_factors=(4, 2, 2, 2), relative_pos_embedding=True):
        super().__init__()

        self.stage1 = StageModule(in_channels=channels, hidden_dimension=hidden_dim, layers=layers[0],
                                  downscaling_factor=downscaling_factors[0], num_heads=heads[0], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage2 = StageModule(in_channels=hidden_dim, hidden_dimension=hidden_dim * 2, layers=layers[1],
                                  downscaling_factor=downscaling_factors[1], num_heads=heads[1], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage3 = StageModule(in_channels=hidden_dim * 2, hidden_dimension=hidden_dim * 4, layers=layers[2],
                                  downscaling_factor=downscaling_factors[2], num_heads=heads[2], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)
        self.stage4 = StageModule(in_channels=hidden_dim * 4, hidden_dimension=hidden_dim * 8, layers=layers[3],
                                  downscaling_factor=downscaling_factors[3], num_heads=heads[3], head_dim=head_dim,
                                  window_size=window_size, relative_pos_embedding=relative_pos_embedding)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 8),
            nn.Linear(hidden_dim * 8, num_classes)
        )

    def forward(self, img):
        x = self.stage1(img)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = x.mean(dim=[2, 3])
        return self.mlp_head(x)


def swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_s(hidden_dim=96, layers=(2, 2, 18, 2), heads=(3, 6, 12, 24), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_b(hidden_dim=128, layers=(2, 2, 18, 2), heads=(4, 8, 16, 32), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)


def swin_l(hidden_dim=192, layers=(2, 2, 18, 2), heads=(6, 12, 24, 48), **kwargs):
    return SwinTransformer(hidden_dim=hidden_dim, layers=layers, heads=heads, **kwargs)

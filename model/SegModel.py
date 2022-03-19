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

import numpy as np
import segmentation_models_pytorch as smp

from .Blocks import *

#============================Segmentation Model Pytorch============================#

class SegModel(nn.Module):
    def __init__(self, n_class=10, in_channel=3, 
                 model_name='Unet', encoder='resnet34', 
                 activation='softmax', init_by_imagenet=False):
        super(SegModel, self).__init__()
        self.available_model_list = ['Unet', 'Unet++', 'MAnet', 
                                     'Linknet', 'FPN', 'PSPnet', 
                                     'PAN', 'DeepLabV3', 'DeepLabV3+']
        self.available_encoder_list = ['resnet18', 'resnet34', 'resnet50', 
                                       'resnet101', 'resnet152', 'resnext50_32x4d',
                                       'resnext101_32x4d', 'resnext101_32x8d', 'resnext101_32x16d',
                                       'resnext101_32x32d', 'resnext101_32x48d', 'timm-resnest14d',
                                       'timm-resnest26d', 'timm-resnest50d',  'timm-resnest101e',
                                       'timm-resnest200e', 'timm-resnest269e', 'timm-resnest50d_4s2x40d',
                                       'timm-resnest50d_1s4x24d', 'timm-res2net50_26w_4s', 'timm-res2net101_26w_4s',
                                       'timm-res2net50_26w_6s', 'timm-res2net50_26w_8s', 'timm-res2net50_48w_2s',
                                       'timm-res2net50_14w_8s', 'timm-res2next50', 'timm-regnetx_002', 
                                       'timm-regnetx_004', 'timm-regnetx_006', 'timm-regnetx_008', 
                                       'timm-regnetx_016', 'timm-regnetx_032', 'timm-regnetx_040',
                                       'timm-regnetx_064', 'timm-regnetx_080', 'timm-regnetx_120',
                                       'timm-regnetx_160', 'timm-regnetx_320', 'timm-regnety_002',
                                       'timm-regnety_004', 'timm-regnety_006', 'timm-regnety_008',
                                       'timm-regnety_016', 'timm-regnety_032', 'timm-regnety_040', 
                                       'timm-regnety_064', 'timm-regnety_080', 'timm-regnety_120',
                                       'timm-regnety_160', 'timm-regnety_320','senet154',
                                       'se_resnet50', 'se_resnet101', 'se_resnet152',
                                       'se_resnext50_32x4d', 'se_resnext101_32x4d', 'timm-skresnet18',
                                       'timm-skresnet34', 'timm-skresnet50_32x4d', 'densenet121',
                                       'densenet169', 'densenet201', 'densenet161',
                                       'inceptionresnetv2', 'inceptionv4', 'xception',
                                       'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',
                                       'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5',
                                       'efficientnet-b6', 'efficientnet-b7', 'timm-efficientnet-b0',
                                       'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3',
                                       'timm-efficientnet-b4', 'timm-efficientnet-b5', 'timm-efficientnet-b6',
                                       'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 
                                       'timm-efficientnet-lite0', 'timm-efficientnet-lite1', 'timm-efficientnet-lite2',
                                       'timm-efficientnet-lite3', 'timm-efficientnet-lite4', 'mobilenet_v2',
                                       'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131', 'vgg11', 'vgg11_bn',
                                       'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
        self.available_activation_list = ['sigmoid', 'softmax', 'logsoftmax', 'tanh', 'identity', None]
        
        self.n_class=n_class
        self.in_channel =in_channel
        
        if model_name not in self.available_model_list:
            raise ValueError('model name should be in the available model list')
        if encoder not in self.available_encoder_list:
            raise ValueError('encoder should be in the available encoder list')
        if activation not in self.available_activation_list:
            raise ValueError('activation should be in the available activation list')
        
        self.model_name = model_name
        self.encoder = encoder
        self.activation = activation
        self.init_weight_way = None
        if init_by_imagenet:
            self.init_weight_way = 'imagenet'
        
        self.initial_model()
        
    def initial_model(self):
        if self.model_name == self.available_model_list[0]:
            self.model = smp.Unet(encoder_name=self.encoder, 
                                  in_channels=self.in_channel,
                                  classes=self.n_class,
                                  activation=self.activation,
                                  decoder_attention_type=None,
                                  encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[1]:
            self.model = smp.UnetPlusPlus(encoder_name=self.encoder,
                                          in_channels=self.in_channel,
                                          classes=self.n_class,
                                          activation=self.activation,
                                          decoder_attention_type=None,
                                          encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[2]:
            self.model = smp.MAnet(encoder_name=self.encoder,
                                   in_channels=self.in_channel,
                                   classes=self.n_class,
                                   activation=self.activation,
                                   encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[3]:
            self.model = smp.Linknet(encoder_name=self.encoder,
                                     in_channels=self.in_channel,
                                     classes=self.n_class,
                                     activation=self.activation,
                                     encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[4]:
            self.model = smp.FPN(encoder_name=self.encoder,
                                 in_channels=self.in_channel,
                                 classes=self.n_class,
                                 activation=self.activation,
                                 encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[5]:
            self.model = smp.PSPNet(encoder_name=self.encoder,
                                    in_channels=self.in_channel,
                                    classes=self.n_class,
                                    activation=self.activation,
                                    encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[6]:
            self.model = smp.PAN(encoder_name=self.encoder,
                                 in_channels=self.in_channel,
                                 classes=self.n_class,
                                 activation=self.activation,
                                 encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[7]:
            self.model = smp.DeepLabV3(encoder_name=self.encoder,
                                       in_channels=self.in_channel,
                                       classes=self.n_class,
                                       activation=self.activation,
                                       encoder_weights=self.init_weight_way)
        elif self.model_name == self.available_model_list[8]:
            self.model = smp.DeepLabV3Plus(encoder_name=self.encoder,
                                           in_channels=self.in_channel,
                                           classes=self.n_class,
                                           activation=self.activation,
                                           encoder_weights=self.init_weight_way)
        else:
            raise ValueError('this is a unavailable model name, please check it.')
        
    def forward(self, x):
        out = self.model(x)
        return out
    
#============================Segmentation Model Pytorch============================#

class test_model(nn.Module):
    def __init__(self, n_class=4):
        super(test_model, self).__init__()
        self.encoder = resnet_down(in_channel=3, out_channel=8, down_factor=2)
        self.att = CBAM_Block(in_channel=8, is_parallel=True)
        self.decoder = resnet_up(in_channel=16, up_factor=2, bilinear=True)
        self.seghead = SegHead(in_channels=8, n_classes=n_class)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.att(out)
        out = self.decoder(out)
        out = self.seghead(out)
        
        return out

class SegTransformer(nn.Module):
    def __init__(self, image_size, patch_size, 
                 num_classes, dim, depth, heads, 
                 mlp_dim, pool = 'cls', channels = 3, 
                 dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        self.vit = VisionTransformerBlock(image_size=image_size, patch_size=patch_size,
                                          num_classes=num_classes, dim=dim, depth=depth, 
                                          heads=heads, mlp_dim=mlp_dim, pool=pool, 
                                          channels=channels, dim_head=dim_head, dropout=dropout, 
                                          emb_dropout=emb_dropout)
        self.seghead = SegMlpHead(dim=dim, n_class=num_classes, image_size=image_size, patch_size=patch_size)
    
    def forward(self, x, mask=None):
        x = self.vit(x, mask=mask)
        out = self.seghead(x)
        
        return out

class SDFormer(nn.Module):
    def __init__(self,
                 n_classes,
                 hidden_dim,
                 layers,
                 heads,
                 in_channels=3,
                 head_dim=32,
                 windows_size=8,
                 downscaling_factors=(2,2,2,2),
                 relative_pos_embedding=True,
                 skip_connect=True):
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

        if isinstance(windows_size, list) or isinstance(windows_size, tuple):
            if len(windows_size)==2 and len(windows_size)!=len(downscaling_factors):
                windows_size = [windows_size for _ in range(len(downscaling_factors))]
            assert len(windows_size)==len(downscaling_factors), 'len(window_size) must equal len(downscaling_factors) !'
        elif isinstance(windows_size, int):
            nws = [windows_size for _ in range(len(downscaling_factors))]
            windows_size = nws
        else:
            TypeError('window_size must be one of (tuple, list, int)')

        
        decoder_heads = heads[::-1]
        decoder_hidden_dim = hidden_dim[::-1]
        up_scale_factors = downscaling_factors[::-1]
        decoder_inchannel = decoder_hidden_dim[0]
        decoder_hidden_dim = decoder_hidden_dim[1:]
        if isinstance(decoder_hidden_dim, tuple):
            new_dim = [i for i in decoder_hidden_dim]
            decoder_hidden_dim = new_dim
        decoder_hidden_dim.append(n_classes)
        decoder_window_size = windows_size[::-1]

        self.encoder = SwinTransformerEncoder(hidden_dim=hidden_dim,
                                              layers=layers,
                                              heads=heads,
                                              channels=in_channels,
                                              head_dim=head_dim,
                                              window_size=windows_size,
                                              downscaling_factors=downscaling_factors,
                                              relative_pos_embedding=relative_pos_embedding)
        self.decoder = SwinTransformerDecoder(inchannels=decoder_inchannel,
                                              hidden_dim=decoder_hidden_dim,
                                              heads=decoder_heads,
                                              head_dim=head_dim,
                                              window_size=decoder_window_size,
                                              upscale_factors=up_scale_factors,
                                              relative_pos_embedding=relative_pos_embedding,
                                              skip_connect=skip_connect)
        self.activation = nn.Softmax2d()
        #self.activation = nn.Softmax(dim=1)
    def forward(self, x):
        features = self.encoder(x)
        #print('features: ',features)
        out = self.decoder(features)
        #print('out: ',out)
        out = self.activation(out)
        return out

class CBAMUnet(nn.Module):
    def __init__(self, n_class, in_channel=3):
        super(CBAMUnet, self).__init__()
        self.down1 = resnet_down(in_channel=in_channel, out_channel=32)
        self.down2 = resnet_down(in_channel=32, out_channel=64)
        self.down3 = resnet_down(in_channel=64, out_channel=128)
        self.down4 = resnet_down(in_channel=128, out_channel=256)
        
        self.cbam1 = CBAM_Block(in_channel=32)
        self.cbam2 = CBAM_Block(in_channel=64)
        self.cbam3 = CBAM_Block(in_channel=128)
        self.cbam4 = CBAM_Block(in_channel=256)
        
        self.up1 = resnet_up(in_channels=64)
        self.up2 = resnet_up(in_channel=128)
        self.up3 = resnet_up(in_channel=256)
        self.up4 = resnet_up(in_channel=256)
        
        self.convconcat1 = BasicConv(in_channels=64, out_channels=32)
        self.convconcat2 = BasicConv(in_channels=128, out_channels=64)
        self.convconcat3 = BasicConv(in_channels=256, out_channels=128)
        
        

    def forward(self, x):
        return x

def build_model(config, inchannel=3, n_class=4, segtask=True):
    if not segtask:
        n_class=3
    if config['name'] == 'Segmodel':
        net = SegModel(n_class=n_class, 
                        in_channel=inchannel, 
                        model_name=config['model_name'],
                        encoder=config['encoder'],
                        activation=config['activation'],
                        init_by_imagenet=config['init_by_imagenet'])
    elif config['name'] == 'Vit':
        net = SegTransformer(image_size=config['image_size'],
                             patch_size=config['patch_size'],
                             num_classes=n_class,
                             dim=config['dim'],
                             depth=config['depth'],
                             heads=config['heads'],
                             mlp_dim=config['mlp_dim'],
                             pool=config['pool'],
                             channels=inchannel,
                             dim_head=config['dim_head'],
                             dropout=config['dropout'],
                             emb_dropout=config['emb_dropout'])
    elif config['name']=='SwinTransformer':
        net = SDFormer(n_classes=n_class, 
                       hidden_dim=config['hidden_dim'], 
                       layers=config['layers'],
                       heads=config['heads'],
                       in_channels=inchannel,
                       head_dim=config['head_dim'],
                       windows_size=config['window_size'],
                       downscaling_factors=config['down_scaling_factors'],
                       relative_pos_embedding=config['relative_pos_embedding'],
                       skip_connect=config['skip_connect'])
    else:
        net = 0
        raise ValueError('the supported config[name] only include Segmodel, Vit, SwinTransformer')

    return net

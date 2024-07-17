import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from transformers import ViTImageProcessor, ViTForImageClassification
from einops.layers.torch import Rearrange
from einops import rearrange
import math
import numpy as np
import timm
from timm.models.layers import to_2tuple, trunc_normal_

class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        self.conv2 = nn.Conv1d(out_dim, out_dim*2, kernel_size, padding="same")
        
        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        X = self.conv2(X)
        X = F.glu(X, dim=-2)

        return self.dropout(X)
    

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.pad, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        if self.pad > 0:
            x = x[:, :, :-self.pad]
        return x

class WaveNetBlock_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, p_drop=0.2):
        super(WaveNetBlock_1, self).__init__()
        self.causal_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p_drop)
        self.conv1x1_residual = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.conv1x1_skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.causal_conv(x)
        out = self.batchnorm(out)
        out = F.tanh(out) * F.sigmoid(out)
        out = self.dropout(out)
        residual = self.conv1x1_residual(out)
        skip = self.conv1x1_skip(out)
        return residual + x, skip

class WaveNet_1(nn.Module):
    def __init__(self, num_classes, num_channels, num_blocks, kernel_size, dilations):
        super(WaveNet_1, self).__init__()
        self.blocks = nn.ModuleList([
            WaveNetBlock_1(num_channels, num_channels, kernel_size, dilation)
            for dilation in dilations
        ])
        self.conv1x1 = nn.Conv1d(num_channels, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        x = sum(skip_connections)
        x = F.gelu(x)
        x = self.conv1x1(x)
        return F.adaptive_avg_pool1d(x, 1).squeeze(-1)

class BasicWaveClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Sequential(WaveBlock(8, 3, 12), WaveBlock())


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.dilated_conv = nn.Conv1d(in_channels=residual_channels,
                                      out_channels=dilation_channels,
                                      kernel_size=kernel_size,
                                      dilation=dilation,
                                      padding="same")
        self.residual_conv = nn.Conv1d(in_channels=dilation_channels,
                                       out_channels=residual_channels,
                                       kernel_size=1,
                                       padding="same")
        self.skip_conv = nn.Conv1d(in_channels=dilation_channels,
                                   out_channels=skip_channels,
                                   kernel_size=1,
                                   padding="same")
        self.gate_conv = nn.Conv1d(in_channels=residual_channels,
                                   out_channels=dilation_channels,
                                   kernel_size=kernel_size,
                                   dilation=dilation,
                                   padding="same")

    def forward(self, x):
        # Gated activation unit
        output = torch.tanh(self.dilated_conv(x)) * torch.sigmoid(self.gate_conv(x))
        # For skip connection
        skip = self.skip_conv(output)
        # For residual connection
        residual = self.residual_conv(output) + x
        return residual, skip

class WaveNet(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, end_channels, kernel_size, num_layers, num_stacks):
        super(WaveNet, self).__init__()
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.num_stacks = num_stacks

        
        self.residual_blocks = nn.ModuleList()
        for s in range(num_stacks):
            for l in range(num_layers):
                self.residual_blocks.append(ResidualBlock(residual_channels, dilation_channels, skip_channels, kernel_size, dilation=2**l))
        
        self.conv_out1 = nn.Conv1d(in_channels=skip_channels, out_channels=end_channels, kernel_size=1)
        self.conv_out2 = nn.Conv1d(in_channels=end_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for layer in self.residual_blocks:
            x, skip = layer(x)
            skip_connections.append(skip)

        out = sum(skip_connections)
        out = F.relu(out)
        out = self.conv_out1(out)
        out = F.relu(out)
        out = self.conv_out2(out)
        return out

class WaveBlock(nn.Module):
    def __init__(self, filters, kernel_size, n):
        super(WaveBlock, self).__init__()
        self.dilation_rates = [2 ** i for i in range(n)]
        self.initial_conv = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, padding='same')
        self.tanh_convs = nn.ModuleList([
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding='same', dilation=rate)
            for rate in self.dilation_rates
        ])
        self.sigmoid_convs = nn.ModuleList([
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding='same', dilation=rate)
            for rate in self.dilation_rates
        ])
        self.final_convs = nn.ModuleList([
            nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=1, padding='same')
            for _ in self.dilation_rates
        ])

    def forward(self, x):
        x = self.initial_conv(x)
        res_x = x
        for tanh_conv, sigmoid_conv, final_conv in zip(self.tanh_convs, self.sigmoid_convs, self.final_convs):
            tanh_out = torch.tanh(tanh_conv(x))
            sigm_out = torch.sigmoid(sigmoid_conv(x))
            x = tanh_out * sigm_out
            x = final_conv(x)
            res_x = res_x + x
        return res_x

    

import torchvision.models as models



class ImageInputModule(nn.Module):
    def __init__(self, hid_dim):
        super(ImageInputModule, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 最後の分類層を除去
        self.fc = nn.Linear(resnet.fc.in_features, hid_dim)  # 潜在空間にマッピング

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # フラットに変換
        x = self.fc(x)
        return x
    

class WaveEncoerModule(nn.Module):
    def __init__(self, input_dims):
        super(WaveEncoerModule, self).__init__()

        self.dims = input_dims
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=dim, out_channels=32, kernel_size=1),
                WaveNet(32, 32, 256, 256, 2, 10, 2),
                #WaveBlock(filters=12 , kernel_size=3, n=12),
                nn.AdaptiveAvgPool1d(1)
            )
            for dim in input_dims
        ])
        efficientnet = models.efficientnet_v2_s(pretrained=False)
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),  # 入力チャンネルを1に変更
            *list(efficientnet.features.children())[1:]
        )
        

    def forward(self, x):
        features = [block(data).squeeze(-1) for block, data in zip(self.blocks, torch.split(x, self.dims, dim=1))] #(128, 12) × 14 <- list
        
        # 特徴を結合
        combined_features = torch.cat(features, dim=1) #(128, 12*14)
        
        # 2D Imageの形に変換
        combined_features = combined_features.view(combined_features.size(0), 1, -1, 1)
        
        # EfficientNet V2 Smallを通す
        x = self.features(combined_features)
        x = x.view(x.size(0), -1)  # フラットに変換
        return x


class CLIP(nn.Module):
    def __init__(self,
                 image_feature_size: int,
                 embed_dim: int,
                 in_channels: int,
                 seq_len: int,
                 MEG_hid_dim: int,
                 ):
        super().__init__()

        dims = [24, 33, 19, 20, 34, 24, 33, 19, 20, 34, 4, 3, 3, 1]

        #image encoder
        self.visual = ImageInputModule(embed_dim)


        #brain wave encoder
        #self.brainwave_encoder = pretrained_model()
        self.MEG_encoder = WaveEncoerModule(dims)
        self.fc = nn.Linear(1280*6, embed_dim)
        #self.MEG_encoder = nn.Sequential(BasicConvClassifier(in_channels=in_channels, hid_dim=MEG_hid_dim), nn.Linear(MEG_hid_dim, embed_dim))

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image_x):
        return self.visual(image_x)
    
    def encode_MEG(self, x):
        return self.fc(self.MEG_encoder(x))

    def forward(self, image, x):
        image_features = self.encode_image(image)
        x_features = self.encode_MEG(x)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logit_per_image = logit_scale * image_features @ x_features.t()
        logit_per_x = logit_per_image.t()

        return logit_per_image, logit_per_x



class FT_model(nn.Module):
    def __init__(self,
                pretrain_path, 
                num_classes,
                image_feature_size: int,
                embed_dim: int,
                in_channels: int,
                seq_len: int,
                MEG_hid_dim: int,
                ):
        super().__init__()
        pretrained_model = CLIP(image_feature_size, embed_dim, in_channels, seq_len, MEG_hid_dim)
        #pretrained_model.load_state_dict(torch.load(pretrain_path))

        self.FT_model = nn.Sequential(pretrained_model.MEG_encoder, nn.Linear(1280, num_classes))

    def forward(self, x):
        return self.FT_model(x)
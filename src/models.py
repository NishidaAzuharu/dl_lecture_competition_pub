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
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     Rearrange("b d 1 -> b d"),
        #     nn.Linear(hid_dim, num_classes),
        # )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        return self.blocks(X)


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
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        
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
    


class BasicWaveClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.blk = nn.Sequential(WaveBlock(8, 3, 12), WaveBlock)




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

    
# class EEGInputModule(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers):
#         super(EEGInputModule, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, 128)  # 128次元の潜在空間にマッピング

#     def forward(self, x):
#         h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = out[:, -1, :]  # 最後のタイムステップの出力を使用
#         out = self.fc(out)
#         return out
    
# class CommonEncoder(nn.Module):
#     def __init__(self):
#         super(CommonEncoder, self).__init__()
#         self.fc = nn.Linear(128, 128)  # 共通のエンコーダ部分

#     def forward(self, x):
#         x = self.fc(x)
#         return x

# class Classifier(nn.Module):
#     def __init__(self, num_classes):
#         super(Classifier, self).__init__()
#         self.fc = nn.Linear(128, num_classes)

#     def forward(self, x):
#         x = self.fc(x)
#         return x

# class MultimodalModel(nn.Module):
#     def __init__(self, eeg_input_size, eeg_hidden_size, eeg_num_layers, num_classes):
#         super(MultimodalModel, self).__init__()
#         self.image_input = ImageInputModule()
#         self.eeg_input = EEGInputModule(eeg_input_size, eeg_hidden_size, eeg_num_layers)
#         self.common_encoder = CommonEncoder()
#         self.classifier = Classifier(num_classes)
    
#     def forward_image(self, image_data):
#         image_features = self.image_input(image_data)
#         encoded_features = self.common_encoder(image_features)
#         class_scores = self.classifier(encoded_features)
#         return class_scores
    
#     def forward_eeg(self, eeg_data):
#         eeg_features = self.eeg_input(eeg_data)
#         encoded_features = self.common_encoder(eeg_features)
#         class_scores = self.classifier(encoded_features)
#         return class_scores
    

class model_3(nn.Module):
    def __init__(self, in_channels, hid_dim, num_classes):
        super().__init__()
        self.input_blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )
        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').vit
        self.enc_bloack = self.model.encoder #in_features:768
        self.layer_norm = self.model.layernorm
        self.classifier_head = nn.Linear(hid_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_blocks(x)
        x = x.permute(0, 2, 1) 
        x = self.enc_bloack(x).last_hidden_state
        x = self.layer_norm(x)
        x = x[:, 0]
        x = self.classifier_head(x)
        return x


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         return self.relu(self.bn(self.conv(x)))

# class model_3(nn.Module):
#     def __init__(self, in_channels, hid_dim, num_classes, device = "cuda"):
#         super().__init__()
#         self.input_blocks = nn.Sequential(
#             ConvBlock(in_channels, hid_dim),
#             ConvBlock(hid_dim, hid_dim),
#         )
#         self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').vit
#         self.enc_block = self.model.encoder
#         self.layer_norm = self.model.layernorm
#         self.classifier_head = nn.Linear(768, num_classes)  # 768はViTの出力次元
        
#     def forward(self, x):
#         x = self.input_blocks(x)
#         # ViTは特定の形状の入力を期待しているため、画像を適切な形状に変換する必要があります。
#         batch_size = x.shape[0]
#         x = x.reshape(batch_size, 3, 224, 224)  # 入力画像の形状にリシェープ
#         x = self.model.embeddings(x)  # ViTの埋め込み層を通過
#         x = self.enc_block(x)
#         x = self.layer_norm(x)
#         x = x[:, 0]  # クラストークンを抽出
#         x = self.classifier_head(x)
#         return x

            
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class pretrained_model(nn.Module):
    def __init__(self, num_classes, channel, seq_len):
        super(pretrained_model, self).__init__()


        timm.models.vision_transformer.PatchEmbed = PatchEmbed
        self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=True)
        self.original_num_patches = self.v.patch_embed.num_patches
        self.original_hw = int(self.original_num_patches ** 0.5)
        self.original_embedding_dim = self.v.pos_embed.shape[2]
        self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, num_classes))
        f_dim, t_dim = self.get_shape(10, 10, channel, seq_len)
        num_patches = f_dim * t_dim
        self.v.patch_embed.num_patches = num_patches

        new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(10, 10))
        new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
        new_proj.bias = self.v.patch_embed.proj.bias
        self.v.patch_embed.proj = new_proj

        
        # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
        new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.original_hw, self.original_hw)
        # cut (from middle) or interpolate the second dimension of the positional embedding
        if t_dim <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, :, int(self.original_hw / 2) - int(t_dim / 2): int(self.original_hw / 2) - int(t_dim / 2) + t_dim]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.original_hw, t_dim), mode='bilinear')
        # cut (from middle) or interpolate the first dimension of the positional embedding
        if f_dim <= self.original_hw:
            new_pos_embed = new_pos_embed[:, :, int(self.original_hw / 2) - int(f_dim / 2): int(self.original_hw / 2) - int(f_dim / 2) + f_dim, :]
        else:
            new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        # flatten the positional embedding
        new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
        # concatenate the above positional embedding with the cls token and distillation token of the deit model.
        self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
        
        # new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
        # self.v.pos_embed = new_pos_embed
        # trunc_normal_(self.v.pos_embed, std=.02)

    
    def get_shape(self, f_stride, t_stride, input_f_dim, input_t_dim):
        test_input = torch.randn(1, 1, input_f_dim, input_t_dim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(f_stride, t_stride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def forward(self, x):
        x = x.unsqueeze(1)
        B = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim = 1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)

        return x
    
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


from typing import Tuple, Union
class CLIP(nn.Module):
    def __init__(self,
                 image_feature_size: int,
                 embed_dim: int,
                 # vision
                 # brain wave
                 in_channels: int,
                 seq_len: int,
                 MEG_hid_dim: int,
                 ):
        super().__init__()

        #image encoder
        self.visual = ImageInputModule(embed_dim)


        #brain wave encoder
        #self.brainwave_encoder = pretrained_model()
        self.MEG_encoder = BasicConvClassifier(in_channels=in_channels, hid_dim=MEG_hid_dim)
        self.fc = nn.Linear(MEG_hid_dim, embed_dim)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image_x):
        return self.visual(image_x.type(self.dtype))
    
    def encode_MEG(self, x):
        x = self.fc(self.MEG_encoder(x))

    def forward(self, image, x):
        image_features = self.encode_image(image)
        x_features = self.encode_MEG(x)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        x_features = x_features / x_features.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logit_per_image = logit_scale * image_features @ x_features.t()
        logit_per_x = logit_per_x.t()

        return logit_per_image, logit_per_x

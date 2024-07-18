import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops.layers.torch import Rearrange
from einops import rearrange


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
        p_drop: float = 0.2,
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

class WaveNet(nn.Module):
    def __init__(self, num_classes, num_channels, num_blocks, kernel_size, dilations):
        super(WaveNet, self).__init__()
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


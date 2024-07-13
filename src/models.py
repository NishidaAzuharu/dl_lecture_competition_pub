import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


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
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")
        
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

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)
    


class ImageInputModule(nn.Module):
    def __init__(self):
        super(ImageInputModule, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # 最後の分類層を除去
        self.fc = nn.Linear(resnet.fc.in_features, 128)  # 128次元の潜在空間にマッピング

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # フラットに変換
        x = self.fc(x)
        return x
    
class EEGInputModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EEGInputModule, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 128)  # 128次元の潜在空間にマッピング

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]  # 最後のタイムステップの出力を使用
        out = self.fc(out)
        return out
    
class CommonEncoder(nn.Module):
    def __init__(self):
        super(CommonEncoder, self).__init__()
        self.fc = nn.Linear(128, 128)  # 共通のエンコーダ部分

    def forward(self, x):
        x = self.fc(x)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x

class MultimodalModel(nn.Module):
    def __init__(self, eeg_input_size, eeg_hidden_size, eeg_num_layers, num_classes):
        super(MultimodalModel, self).__init__()
        self.image_input = ImageInputModule()
        self.eeg_input = EEGInputModule(eeg_input_size, eeg_hidden_size, eeg_num_layers)
        self.common_encoder = CommonEncoder()
        self.classifier = Classifier(num_classes)
    
    def forward_image(self, image_data):
        image_features = self.image_input(image_data)
        encoded_features = self.common_encoder(image_features)
        class_scores = self.classifier(encoded_features)
        return class_scores
    
    def forward_eeg(self, eeg_data):
        eeg_features = self.eeg_input(eeg_data)
        encoded_features = self.common_encoder(eeg_features)
        class_scores = self.classifier(encoded_features)
        return class_scores
import os, sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import BasicConvClassifier
from src.utils import set_seed
import mne
"""
installation

pip install -U mne
pip install cupy
MNE_USE_CUDA=true python -c "import mne; mne.cuda.init_cuda(verbose=True)"
mne.utils.set_config('MNE_USE_CUDA', 'true')  
# mne.io.Raw.filter()などにn_jobs='cuda'を渡す
"""

@hydra.main(version_base=None, config_path="configs", config_name="config") #configfileの指定
def run(args: DictConfig):
    set_seed(args.seed)

    data_dir = args.data_dir
    split = "train"
    X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
    y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
    subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

    # print(X[0])
    # print(subject_idxs[10])
    # data_np = X.numpy()

    # fig, axes = plt.subplots(3, 1, figsize=(15, 100))  # 271行1列のサブプロット、サイズ調整


    # for i in range(3):
    #     axes[i].plot(data_np[0][i])
    #     axes[i].set_title(f'Channel {i+1}')
    #     axes[i].set_xlabel('Time')
    #     axes[i].set_ylabel('Amplitude')

    # plt.tight_layout()
    # plt.show()

        



if __name__ == "__main__":
    run()

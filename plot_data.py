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
from mne.channels import make_standard_montage
from mne.datasets.brainstorm import bst_auditory
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings("ignore")

# MNEのロギングレベルを設定
mne.set_log_level("WARNING")
# mne.utils.set_config('MNE_USE_CUDA', 'true') 

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
    splits = ["train", "val"]
    for split in splits:
        X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        # y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
        # subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        np_X = X.numpy()
        n_channels = 271
        n_samples = 281
        sfreq = 200  # サンプリング周波数

        ch_names  = [f'MEG {i}' for i in range(n_channels)]  # 275チャンネルの名前を作成
        montage = mne.channels.make_standard_montage('biosemi64')  # 標準のモンタージュを使用
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='mag')
        ica = ICA(n_components=20, random_state=97, max_iter=800)


        output = []
        for data in tqdm(np_X):
            raw = mne.io.RawArray(data, info)

            raw.set_montage(montage, match_case=False)

            ica.fit(raw, verbose=False)
            ica.exclude = [0, 1]
            raw_corrected = ica.apply(raw.copy(), verbose=False)
            corrected_data = raw_corrected.get_data()
            corrected_data = torch.tensor(corrected_data)
            output.append(corrected_data)

        corrected_tensor = torch.tensor(np.array(output), dtype=torch.float32)
        torch.save(corrected_tensor, f'data/ICA_{split}_X.pt')


    # 修正後のデータをプロット
    # raw.plot(title="Original Data")
    # raw_corrected.plot(title="ICA Corrected Data")
    # plt.show()





    """
    #MEGのサンプルデータを読み込んでinfoを作成
    data_path = bst_auditory.data_path()
    subject = "bst_auditory"
    info = mne.create_info(ch_names=[], sfreq=1000)  # 仮のサンプリングレート
    info = mne.io.read_raw_ctf(data_path / "MEG" / subject / "S01_AEF_20131218_01.ds", preload=False).info
    channel_names = [sensor for sensor in info['ch_names'] if sensor.startswith('M')]
    exclude_channels = ['MLF25-4408', 'MRF43-4408', 'MRO13-4408', 'MRO11-4408']
    channel_names = [ch for ch in channel_names if ch not in exclude_channels]
    info = mne.create_info(ch_names=channel_names, sfreq=200, ch_types='grad')
    """

    # montage = mne.channels.make_standard_montage('standard_1020')
    # info.set_montage(montage)
    # raw = mne.io.RawArray(np_X[0], info)

    


    # # #raw.plot(picks=['MEG 001'])
    # # raw.plot(duration=30, n_channels=4, scalings={'meg': 200e-6})
    # # plt.show()

    # start, stop = raw.time_as_index([0, 1])
    # plt.figure()
    # plt.plot(raw.times[start:stop], raw.get_data(picks=['MEG 001'])[0, start:stop])
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('MEG 001 Channel')
    # plt.show()



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

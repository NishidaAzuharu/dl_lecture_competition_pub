import os
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils import set_seed
import mne
from mne.preprocessing import ICA
import warnings
warnings.filterwarnings("ignore")

# MNEのロギングレベルを設定
mne.set_log_level("WARNING")
# mne.utils.set_config('MNE_USE_CUDA', 'true') 


@hydra.main(version_base=None, config_path="configs", config_name="config") #configfileの指定
def run(args: DictConfig):
    set_seed(args.seed)

    data_dir = args.data_dir
    splits = ["train", "val", "test"]
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

if __name__ == "__main__":
    run()

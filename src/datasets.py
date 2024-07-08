import os
import numpy as np
import torch
from torchvision import transforms
from typing import Tuple
from termcolor import cprint
from PIL import Image



class ThingsIMGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, image_size: int, data_dir: str = "images") -> None:
        super().__init__()

        assert split in ["train", "val"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        self.image_size = image_size

        path_list = []
        with open(os.path.join(data_dir, f"{split}_image_paths.txt"), 'r', encoding='utf-8') as file:
            for line in file:
                path = os.path.join(data_dir, "images", line.strip())
                path_list.append(path)

        self.img_path = path_list
        self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
        assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.y)
    
    def __getitem__(self, i):
        img = Image.open(self.img_path[i])
        img_array = np.array(img).reshape((self.image_size, self.image_size)).flatten()
        return self.transform(img_array), self.y[i]
    
    @property
    def num_channels(self) -> int:
        return 3 #RGB
    
    @property
    def image_size(self) -> int:
        return self.image_size * self.image_size



class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data") -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
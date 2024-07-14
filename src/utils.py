import random
import numpy as np
import torch

def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_lr(ooptimizer):
    for param_group in ooptimizer.param_groups:
        return param_group["lr"]
    

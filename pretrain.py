import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm
import torch.nn as nn

from src.datasets import ThingsPretrainDataset
from src.models import BasicConvClassifier, CLIP
from src.utils import set_seed, CosineScheduler, set_lr


config = {
    "image_size": [32, 32],
    "patch_size": [2, 2],
    "emb_dim": 192,
    "enc_layers": 12,
    "enc_heads": 3,
    "enc_dim_head": 64,
    "enc_mlp_dim": 192,
    "dec_layers": 4,
    "dec_heads": 3,
    "dec_dim_head": 64,
    "dec_mlp_dim": 192,
    "mask_ratio": 0.75,
    "dropout": 0.
}



@hydra.main(version_base=None, config_path="configs", config_name="config") #configfileの指定
def run(args: DictConfig):
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    
    train_set = ThingsPretrainDataset("train", args.image_size, args.data_dir)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsPretrainDataset("val", args.image_size, args.data_dir)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    

    # ------------------
    #       Model
    # ------------------
    model = CLIP(args.image_size, 768, train_set.num_channels, train_set.seq_len, 128).to(args.device)


    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98),eps=1e-6, weight_decay=0.2)
    warmup_length = 10
    scheduler = CosineScheduler(args.pretrain_epochs, args.lr, warmup_length)

    # ------------------
    #   Start training
    # ------------------  

    min_val_loss = 100000
    grad_clip = 0.1
    loss = nn.CrossEntropyLoss()
    for epoch in range(args.pretrain_epochs):
        print(f"Epoch {epoch+1}/{args.pretrain_epochs}")
        new_lr = scheduler(epoch)
        set_lr(new_lr, optimizer)
        
        train_loss, val_loss = [], []
        
        model.train()
        for img, X in tqdm(train_loader, desc="Train"):
            img, X = img.to(args.device), X.to(args.device)

            logit_img, logit_X = model(img, X)

            gt = torch.arange(len(img), dtype=torch.long, device=args.device)

            total_loss = (loss(logit_img, gt) + loss(logit_X, gt)) / 2

            train_loss.append(total_loss.item())
            
            total_loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        for img, X in tqdm(val_loader, desc="Validation"):
            img, X = img.to(args.device), X.to(args.device)
            
            with torch.no_grad():
                logit_img, logit_X = model(img, X)

            gt = torch.arange(len(img), dtype=torch.long, device=args.device)
            
            loss_ = (loss(logit_img, gt) + loss(logit_X, gt)) / 2
            val_loss.append(loss_.item())

        print(f"Epoch {epoch+1}/{args.pretrain_epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "pretrained_model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "val_loss": np.mean(val_loss)})
        
        if np.mean(val_loss) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "pretrained_model_best.pt"))
            min_val_loss = np.mean(val_loss)
            

if __name__ == "__main__":
    run()

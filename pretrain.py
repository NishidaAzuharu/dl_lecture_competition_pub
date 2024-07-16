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
from src.utils import set_seed


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
    model = CLIP(**config).to(args.device)


    # ------------------
    #     Optimizer
    # ------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.98),eps=1e-6, weight_decay=0.2)

    # ------------------
    #   Start training
    # ------------------  
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    loss = nn.CrossEntropyLoss()
    for epoch in range(args.pretrain_epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc, val_loss, val_acc = [], [], [], []
        
        model.train()
        for img, X in tqdm(train_loader, desc="Train"):
            optimizer.zero_grad()
            img, X = X.to(args.device), img.to(args.device)

            logit_img, logit_X = model(img, X)

            gt = torch.arange(len(img), dtype=torch.long, device=args.device)

            total_loss = (loss(logit_img, gt) + loss(logit_X, gt)) / 2
            
            total_loss.backward()
            optimizer.step()

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)
            
            with torch.no_grad():
                y_pred = model(X)
            
            val_loss.append(F.cross_entropy(y_pred, y).item())
            val_acc.append(accuracy(y_pred, y).item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})
        
        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)
            
    
    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()

import numpy as np
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import tqdm

import torch
import torchvision
import torchvision.transforms as transforms
from vit import ViT

import matplotlib.pyplot as plt

import hydra
import wandb
from datetime import datetime

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def select_two_classes_from_cifar10(dataset, classes):
    idx = (np.array(dataset.targets) == classes[0]) | (np.array(dataset.targets) == classes[1])
    dataset.targets = np.array(dataset.targets)[idx]
    dataset.targets[dataset.targets==classes[0]] = 0
    dataset.targets[dataset.targets==classes[1]] = 1
    dataset.targets= dataset.targets.tolist()  
    dataset.data = dataset.data[idx]
    return dataset

def prepare_dataloaders(batch_size, classes=[3, 7]):
    # TASK: Experiment with data augmentation
    train_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=test_transform)

    # select two classes 
    trainset = select_two_classes_from_cifar10(trainset, classes=classes)
    testset = select_two_classes_from_cifar10(testset, classes=classes)

    # reduce dataset size
    trainset, _ = torch.utils.data.random_split(trainset, [5000, 5000])
    testset, _ = torch.utils.data.random_split(testset, [1000, 1000])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True
    )
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False
    )
    return trainloader, testloader, trainset, testset

@hydra.main(version_base=None, config_path="config", config_name="default_config.yaml")
def main(cfg):
    date_time = datetime.now().strftime("%Y%m%d_%H%M")

    # Read hyperparameters for experiment
    hparams = cfg.experiment
    image_size = hparams["image_size"]
    patch_size = hparams["patch_size"]
    channels = hparams["channels"]
    embed_dim = hparams["embed_dim"]
    num_heads = hparams["num_heads"]
    num_layers = hparams["num_layers"]
    num_classes = hparams["num_classes"]
    pos_enc = hparams["pos_enc"]
    pool = hparams["pool"]
    dropout = hparams["dropout"]
    fc_dim = hparams["fc_dim"]
    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]
    warmup_steps = hparams["warmup_steps"]
    weight_decay = hparams["weight_decay"]
    gradient_clipping = hparams["gradient_clipping"]

    # ✨ W&B: setup
    wandb_cfg = {
        "image_size": image_size,
        "patch_size": patch_size,
        "channels": channels,
        "embed_dim": embed_dim,
        "num_classes": num_classes,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "num_epochs": num_epochs,
        "pos_enc": pos_enc,
        "pool": pool,
        "dropout": dropout,
        "fc_dim": fc_dim,
        "learning_rate": lr,
        "batch_size": batch_size,
        "warmup_steps": warmup_steps,
        "weight_decay": weight_decay,
        "gradient_clipping": gradient_clipping,
    }
    wandb.init(
        project="ex-2",
        entity="adlcv",
        config=wandb_cfg,
        job_type="train",
        name="train_" + date_time,
        dir="./outputs",
    )

    loss_function = nn.CrossEntropyLoss()

    train_iter, test_iter, _, _ = prepare_dataloaders(batch_size=batch_size)

    model = ViT(image_size=image_size, patch_size=patch_size, channels=channels, 
                embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
                pos_enc=pos_enc, pool=pool, dropout=dropout, fc_dim=fc_dim, 
                num_classes=num_classes
    )

    if torch.cuda.is_available():
        model = model.to('cuda')

    opt = torch.optim.AdamW(lr=lr, params=model.parameters(), weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(i / warmup_steps, 1.0))

    # training loop
    best_val_loss = 1e10
    for e in range(num_epochs):
        print(f'\n epoch {e+1}')
        model.train()
        train_loss = 0
        for image, label in tqdm.tqdm(train_iter):
            if torch.cuda.is_available():
                image, label = image.to('cuda'), label.to('cuda')
            opt.zero_grad()
            out = model(image)
            loss = loss_function(out, label)
            loss.backward()
            train_loss += loss.item()
            # if the total gradient vector has a length > 1, we clip it back down to 1.
            if gradient_clipping > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            opt.step()
            sch.step()

        train_loss/=len(train_iter)

        val_loss = 0
        with torch.no_grad():
            model.eval()
            tot, cor= 0.0, 0.0
            for image, label in test_iter:
                if torch.cuda.is_available():
                    image, label = image.to('cuda'), label.to('cuda')
                out = model(image)
                loss = loss_function(out, label)
                val_loss += loss.item()
                out = out.argmax(dim=1)
                tot += float(image.size(0))
                cor += float((label == out).sum().item())
            acc = cor / tot
            val_loss /= len(test_iter)
            print(f'-- train loss {train_loss:.3f} -- validation accuracy {acc:.3f} -- validation loss: {val_loss:.3f}')
            wandb.log({"train_loss": train_loss})
            wandb.log({"validation_accuracy": acc})
            wandb.log({"validation_loss": val_loss})         
            if val_loss <= best_val_loss:
                torch.save(model.state_dict(), 'model.pth')
                best_val_loss = val_loss


if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"]= str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print(f"Model will run on {device}")
    set_seed(seed=1)
    main()

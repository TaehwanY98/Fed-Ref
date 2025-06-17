from CustomDataset import *
from parser import Centralparser
import numpy as np
import warnings
import random
import os
from tqdm import tqdm
import pandas as pd
from torch import nn, int32, int64, float32, save
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
import torch
from torch.nn.functional import one_hot
import sys
if __name__=="__main__":
    sys.path.append("../Network")
    from Resnet import ResNet
else:
    from Network.Resnet import ResNet
from torchmetrics.classification import Accuracy, F1Score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        pass

def train(net, train_loader, valid_loader, epoch, lossf, optimizer, DEVICE, save_path):
    history = {}
    for e in range(epoch):
        net.train()
        for sample in tqdm(train_loader):
            X= torch.stack([s[0] for s in sample], 0)
            Y= torch.Tensor([s[1] for s in sample])
            out = net(X.type(float32).to(DEVICE))
            loss = lossf(out.type(float32).to(DEVICE), one_hot(Y.type(int64), 10).type(float32).squeeze().to(DEVICE))
            loss.backward()
            optimizer.step()          
            optimizer.zero_grad()
        
        if valid_loader != None:
            # print("valid start")
            with torch.no_grad():
                for key, value in valid(net, valid_loader, e, lossf, DEVICE, True).items():
                    if e == 0:
                        history[key] = []
                    history[key].append(value)
        if save_path is not None:            
            save(net.state_dict(), f"./Models/{save_path}/net.pt")
    if valid_loader is not None:                    
        return history
    else:
        return None

def valid(net, valid_loader, e, lossf, DEVICE, Central=False):
    net.eval()
    Dicenary = {'accuracy':0, 'f1score':0}
    length = len(valid_loader) 
    losses = 0
    accf = Accuracy("multiclass", num_classes=10, multidim_average="global").to(DEVICE)
    f1scoref = F1Score("multiclass", num_classes=10, multidim_average="global").to(DEVICE)
    for sample in tqdm(valid_loader, desc="Validation: "):
    
        X= torch.stack([s[0] for s in sample], 0)
        Y= torch.Tensor([s[1] for s in sample])
    
        out = net(X.type(float32).to(DEVICE)) 

        losses += lossf(out.type(float32).to(DEVICE), one_hot(Y.type(int64), 10).type(float32).squeeze().to(DEVICE)).item()
        
        Dicenary[f"accuracy"] += accf(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE)).item()
        Dicenary[f"f1score"] += f1scoref(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE)).item()

    # if Central:
        # logger.info(f"Result epoch {e+1}: loss:{losses/length} accuracy: {Dicenary["accuracy"]/length: .4f} f1score: {Dicenary["f1score"]/length: .4f}")
        
    return {"loss":losses/length, 'accuracy': Dicenary["accuracy"]/length , "f1score":Dicenary["f1score"]/length}


def set_seeds(seed:int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def to_csv(history, file_name):
    pd.DataFrame(history, index=None).to_csv(f"./Csv/{file_name}.csv")


def main():
    warnings.filterwarnings("ignore")
    print("==== Centralized Learning ====")
    args = Centralparser()
    
    make_model_folder(f"./Models/{args.version}")
    set_seeds(args.seed)

    
    net = ResNet()
    if args.pretrained:
        net.load_state_dict(torch.load(f"./Models/{args.version}/net.pt"))
    
    net.to(DEVICE)
    dataset = OCTDL(args.data_dir)
    lossf = nn.BCEWithLogitsLoss(dataset.label_weight)
    lossf.to(DEVICE)
    optimizer = SGD(net.parameters(), lr = args.lr)
    print("==== 모델 아키텍처 ====")
    print(net)
    print("==== Loss ====")
    print(lossf.__class__.__name__)
    print("==== Args ====")
    print(f"seed value: {args.seed}")
    print(f"epoch number: {args.epoch}")
    
    train_dataset, valid_dataset = random_split(dataset, [0.9,0.1], torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=lambda x:x)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=lambda x:x)
    print("==== Training ====")
    history = train(net, train_loader, valid_loader, args.epoch, lossf, optimizer, DEVICE, args.version)
    to_csv(history, args.version)

if __name__=="__main__":
    main()
    
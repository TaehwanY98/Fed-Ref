import getpass
import sys
user = getpass.getuser()
sys.path.append(f"/home/{user}/MICCAI/Network")
import segmentation_models_pytorch as smp
from Network.pytorch3dunet.unet3d.losses import DiceLoss
from Network.Unet import Custom3DUnet
from utils.CustomDataset import Fets2022
from utils.parser import Centralparser
import numpy as np
import warnings
import random
import os
from tqdm import tqdm
import pandas as pd
from torch import nn, int64,float32, save
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD
import torch
import logging
from torch.nn.functional import one_hot

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler('logs.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

diceLoss = smp.losses.DiceLoss(
   mode="multiclass",          # For multi-class segmentation
   classes=None,               # Compute the loss for all classes
   log_loss=False,             # Do not use log version of Dice loss
   from_logits=True,           # Model outputs are raw logits
   smooth=1e-5,                # A small smoothing factor for stability
   ignore_index=None,          # Don't ignore any classes
   eps=1e-7                    # Epsilon for numerical stability
)

focalLoss = smp.losses.FocalLoss(
   mode="multiclass",          # Multi-class segmentation
   alpha=0.1,                 # class weighting to deal with class imbalance
   gamma=4.5,                   # Focusing parameter for hard-to-classify examples
   normalized=True
)

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
            X= torch.stack([s["x"] for s in sample], 0)
            Y= torch.stack([s["y"] for s in sample], 0)
            out = net(X.type(float32).unsqueeze(0).to(DEVICE))
            
            loss = lossf(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE))
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

class CustomHF95(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        # x = torch.softmax(x, dim=0)
        result = 0.0
        for i in self.num_classes:
            result += Hausdolf95().to(DEVICE)(x[i,...], y[i,...]).item()
        return  result/len(self.num_classes)
        
class CustomDice(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        self.num_classes = num_classes
        super().__init__(*args, **kwargs)
    
    def forward(self, x, y):
        result = 0.0
        for i in self.num_classes:
            result += (1-DiceLoss().to(DEVICE)(x[i,...], y[i,...])).item()
        return  result/len(self.num_classes)

class CustomBCEDiceLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.7, beta=0.3, *args, **kwargs):
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        super().__init__(*args, **kwargs)
    
    def forward(self, x, y):
        result = 0.0
        for i in self.num_classes:
            result += (self.beta*diceLoss.to(DEVICE)(x[i,...], y[i,...]) + self.alpha*torch.nn.BCEWithLogitsLoss().to(DEVICE)(x[i,...], y[i,...]))
        return  result
    
class CustomFocalDiceLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def forward(self, x, y):
        return diceLoss.to(DEVICE)(x, y) + focalLoss.to(DEVICE)(x, y)

def valid(net, valid_loader, e, lossf, DEVICE, Central=False):
    net.eval()
    Dicenary = {'mDice':0, 'mHF95':0}
    length = len(valid_loader) 
    losses = 0
    dicef= diceLoss.to(DEVICE)
    hf95f = CustomHF95([range(4)]).to(DEVICE)
    for sample in tqdm(valid_loader, desc="Validation: "):
    
        X= torch.stack([s["x"] for s in sample], 0)
        Y= torch.stack([s["y"] for s in sample], 0)
    
        out = net(X.type(float32).unsqueeze(0).to(DEVICE)) 
    
        losses += lossf(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE)).item()
        Dicenary[f"mDice"] += (1-dicef(out.type(float32).to(DEVICE), Y.type(int64).to(DEVICE))).item()
        Dicenary[f"mHF95"] += hf95f(out.squeeze().type(float32).to(DEVICE), one_hot(Y.type(int64).squeeze(), 4).permute(3, 0, 1, 2).type(float32).to(DEVICE))

    if Central:
        logger.info(f"Result epoch {e+1}: loss:{losses/length} mDice: {Dicenary["mDice"]/length: .4f} HF95: {Dicenary["mHF95"]/length: .4f}")
        
    return {"loss":losses/length, 'mDice': Dicenary["mDice"]/length,'mHF95': Dicenary["mHF95"]/length}


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

class Hausdolf95(nn.Module):
    def __init__(self):
        super(Hausdolf95, self).__init__()
        self.distancef = nn.PairwiseDistance()
    def distance(self, x, y):
        return self.distancef(x, y)
        
    def forward(self, x, y):
        maximum_d=torch.max(self.distance(x, y))
        return maximum_d

def main():
    warnings.filterwarnings("ignore")
    print("==== Centralized Learning ====")
    args = Centralparser()
    
    make_model_folder(f"./Models/{args.version}")
    set_seeds(args.seed)

    
    net = Custom3DUnet(1, 4, True, f_maps=4, layer_order="gcr", num_groups=4)
    if args.pretrained:
        net.load_state_dict(torch.load(f"./Models/{args.version}/net.pt"))
    
    net.to(DEVICE)
    lossf = CustomFocalDiceLoss()
    lossf.to(DEVICE)
    optimizer = SGD(net.parameters(), lr = args.lr)
    print("==== 모델 아키텍처 ====")
    print(net)
    print("==== Loss ====")
    print(lossf.__class__.__name__)
    print("==== Args ====")
    print(f"seed value: {args.seed}")
    print(f"epoch number: {args.epoch}")
    fets2021 = Fets2022(args.data_dir)
    train_dataset, valid_dataset = random_split(fets2021,[0.9,0.1], torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, collate_fn=lambda x:x)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False, collate_fn=lambda x:x)
    print("==== Training ====")
    history = train(net, train_loader, valid_loader, args.epoch, lossf, optimizer, DEVICE, args.version)
    to_csv(history, args.version)

if __name__=="__main__":
    main()
    
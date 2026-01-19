from CustomDataset import *
import numpy as np
import random
import os
from tqdm import tqdm
import pandas as pd
from torch import nn, int32, int64, float32, save
import torch
# from torchmetrics.classification import Accuracy, F1Score
from sklearn.metrics import accuracy_score, f1_score, precision_score

from torch.nn.functional import one_hot
from PIL import Image
import io
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
            X= torch.stack([torch.Tensor(np.array(s.convert("L"))) for s in sample["image"]], 0)
            Y= torch.Tensor(sample["character"])
            if len(sample) != 1:
                out = net(X.unsqueeze(-1).permute(0,3,1,2).to(DEVICE))
                Y = one_hot(Y.type(int64), 62).type(float32)
                loss = lossf(out.squeeze().type(float32).to(DEVICE), Y.type(float32).to(DEVICE))
            else:
                out = net(X.unsqueeze(-1).permute(0,3,1,2).unsqueeze(0).to(DEVICE))
                Y = one_hot(Y.type(int64), 62).type(float32)
                loss = lossf(out.squeeze().type(float32).to(DEVICE), Y.type(float32).to(DEVICE))
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
        return {"loss": loss.item()}

def valid(net, valid_loader, e, lossf, DEVICE, Central=False):
    net.eval()
    Dicenary = {'accuracy':0, 'f1score':0, 'precision':0}
    length = 0
    losses = 0

    for sample in tqdm(valid_loader, desc="Validation: "):
        
        X= torch.stack([torch.Tensor(np.array(s.convert("L"))) for s in sample["image"]], 0)
        Y= torch.Tensor(sample["character"])
        if len(sample) != 1:
            out = net(X.unsqueeze(-1).permute(0,3,1,2).to(DEVICE)) 
            out = out.squeeze()
            Y = one_hot(Y.type(int64), 62).type(float32)
            losses += lossf(out.squeeze().type(float32).to(DEVICE), Y.type(float32).to(DEVICE)).item()
        else:
            out = net(X.unsqueeze(-1).permute(0,3,1,2).unsqueeze(0).to(DEVICE))
            out = out.squeeze()
            Y = one_hot(Y.type(int64), 62).type(float32)
            losses += lossf(out.type(float32).to(DEVICE), Y.type(float32).to(DEVICE)).item()
          
        out = out.softmax(1).argmax(1)
        Dicenary[f"accuracy"] += accuracy_score(out.cpu().detach().numpy(), Y.squeeze().argmax(1).type(int64).cpu().detach().numpy())
        Dicenary[f"f1score"] += f1_score(out.cpu().detach().numpy(), Y.squeeze().argmax(1).type(int64).cpu().detach().numpy(), average="weighted")
        Dicenary[f"precision"] += precision_score(out.cpu().detach().numpy(), Y.squeeze().argmax(1).type(int64).cpu().detach().numpy(), average="weighted")
        # Dicenary[f"jaccard"] += jaccard_score(one_hot(out, 62).cpu().detach().numpy(), Y.squeeze().type(float32).cpu().detach().numpy(), average="weighted")
        # Dicenary[f"hamming"] += hamming_loss(one_hot(out, 62).cpu().detach().numpy(), Y.squeeze().type(float32).cpu().detach().numpy())  
    
        length += 1
    # if Central:
        # logger.info(f"Result epoch {e+1}: loss:{losses/length} accuracy: {Dicenary["accuracy"]/length: .4f} f1score: {Dicenary["f1score"]/length: .4f}")
        
    return {"loss":losses/length, 'accuracy': Dicenary["accuracy"]/length , "f1score":Dicenary["f1score"]/length, "precision":Dicenary["precision"]/length}
    # return {"loss": losses/length}

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

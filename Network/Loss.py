from torch import nn
import torch
import torch.nn.functional as F
class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=0.5, beta=0.5, gamma=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        ce_loss = nn.BCEWithLogitsLoss()(inputs, targets)

        pt = torch.exp(-ce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * ce_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
        
class AsymmetricLoss(nn.Module):
    def __init__(
        self,
            weight = None,
            gamma_pos: float = 0.0,
            gamma_neg: float = 1.0,
            margin: float = 0.2,
            eps: float = 1e-6,
        ):
            super().__init__()
            self.weight = weight
            self.gamma_pos = gamma_pos
            self.gamma_neg = gamma_neg
            self.margin = margin
            self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        p_neg = torch.clamp(torch.sigmoid(y_pred) - self.margin, min=self.eps)
        logit = y_pred * y_true + (torch.log(p_neg) - torch.log(1 - p_neg)) * (1 - y_true)
        bce_loss = nn.BCEWithLogitsLoss()(
            logit, y_true
        )
        p_t = torch.exp(-bce_loss)
        gamma = self.gamma_pos * y_true + self.gamma_neg * (1 - y_true)
        loss = bce_loss * ((1 - p_t) ** gamma)
        return loss.mean()
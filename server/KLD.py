import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, predict, target):
        return self.kld_loss(target.float(), predict.float())

    def kld_loss(self, target: torch.Tensor, predict: torch.Tensor, is_logits: bool = True) -> torch.Tensor:
        """
        Computes KL Divergence Loss: D_KL(target || predict)
        
        Args:
            target: Ground truth probability distribution (or soft targets).
            predict: Model output (logits by default, or raw probabilities).
            is_logits: If True, applies log_softmax to `predict`.
                    If False, assumes `predict` is probabilities and applies torch.log.
        
        Returns:
            Scalar tensor containing the batch mean KL divergence.
        """
        if is_logits:
            log_preds = F.log_softmax(predict, dim=-1)
        else:
            # Clamp to avoid log(0) numerical instability
            eps = 1e-8
            log_preds = torch.log(torch.clamp(predict, min=eps))

        kl_fn = nn.KLDivLoss(reduction='batchmean')
        return kl_fn(log_preds, target)
    
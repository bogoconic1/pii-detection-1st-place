from torch import nn
import torch.nn.functional as F
import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss
import numpy as np

class FocalLoss(nn.Module):
        def __init__(self, gamma=3#2
                     , weight=None, ignore_index=-100):
            super(FocalLoss, self).__init__()
            self.gamma = gamma
            self.weight = weight
            self.ignore_index = ignore_index

        def forward(self, input, target):
            """
            input: [N, C]
            target: [N, ]
            """
            logpt = F.log_softmax(input, dim=1)
            pt = torch.exp(logpt)
            logpt = (1-pt)**self.gamma * logpt
            loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
            return loss


'''
Code taken and modified from:
https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py
'''
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class JaccardLoss(_Loss):
    def __init__(
        self,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases

        Args:
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)

        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(JaccardLoss, self).__init__()

        self.ignore_index = -100
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        y_pred = y_pred[y_true!=self.ignore_index]
        y_true = y_true[y_true!=self.ignore_index]

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = y_pred.log_softmax(dim=-1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(-1)
        dims = (0)

        y_true = F.one_hot(y_true, num_classes)  # N*1536 -> N*1536, 15

        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        return loss.mean()

def soft_jaccard_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score
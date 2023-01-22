import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import DiceScore


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.dice_score = DiceScore()

    def forward(self, y_pred, y_true):
        """Forward pass.
        Args:
        -----
            y_pred (torch.Tensor): The predicted tensor.
            y_true (torch.Tensor): The target tensor.
        Returns:
        --------
            torch.Tensor: The dice loss.
        """
        return 1 - self.dice_score(y_pred, y_true)


class DiceBCELoss(nn.Module):
    """Dice and BCE loss for segmentation.
    from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """

    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()

    def forward(self, inputs, targets, smooth=1):
        """Forward pass.
        Args:
        -----
            inputs (torch.Tensor): The input tensor.
            targets (torch.Tensor): The target tensor.
            smooth (float): Smoothing factor.
        Returns:
        --------
            torch.Tensor: The dice and BCE loss.
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        dice_loss = self.dice_loss(inputs, targets)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


ALPHA = 0.8
GAMMA = 2


class FocalLoss(nn.Module):
    """Focal loss for segmentation.
    from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """

    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        """Forward pass.
        Args:
        -----
            inputs (torch.Tensor): The input tensor.
            targets (torch.Tensor): The target tensor.
            alpha (float): Weighting factor.
            gamma (float): Focusing parameter.
            smooth (float): Smoothing factor.
        Returns:
        --------
            torch.Tensor: The focal loss.
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1 - BCE_EXP) ** gamma * BCE

        return focal_loss

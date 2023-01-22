import numpy as np
import torch
import torch.nn as nn

from monai.metrics.utils import get_mask_edges, get_surface_distance

EPS = 1e-8


class DiceScore(nn.Module):
    """Dice score metric."""

    def __init__(self, weight=None, size_average=True):
        super(DiceScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """Forward pass.
        Args:
        -----
            inputs (torch.Tensor): The input tensor.
            targets (torch.Tensor): The target tensor.
            smooth (float): Smoothing factor.
        Returns:
        --------
            torch.Tensor: The dice score.
        """
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return dice


def compute_hausdorff_monai(pred, gt, max_dist):
    """Computes the Hausdorff distance between two binary masks.
    from: https://www.kaggle.com/code/yiheng/50-times-faster-way-get-hausdorff-with-monai
    Args:
    -----
        pred (torch.Tensor): The predicted tensor.
        gt (torch.Tensor): The target tensor.
        max_dist (float): The maximum distance.
    Returns:
    --------
        float: The Hausdorff distance.
    """
    if np.all(pred == gt):
        return 0.0
    (edges_pred, edges_gt) = get_mask_edges(pred, gt)
    surface_distance = get_surface_distance(edges_pred, edges_gt, distance_metric="euclidean")
    if surface_distance.shape == (0,):
        return 0.0
    dist = surface_distance.max()
    if dist > max_dist:
        return 1.0
    return dist / max_dist

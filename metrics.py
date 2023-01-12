import numpy as np
import torch

EPS = 1e-8


def PSNR(input, target):
    psnr = -10*torch.log10(torch.mean((input - target) ** 2, dim=[1, 2, 3])+EPS)
    return psnr.numpy()
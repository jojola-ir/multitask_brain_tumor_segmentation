import numpy as np
import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image

from dataloader import transformations
from model import build_model


def visualize(image, model, classification=True):
    """Visualize image and its corresponding predicted mask.
    Args:
    -----
        image: (PIL.Image): The image for prediction.
        model: (torch.nn.Module): The model to use for prediction.
    """
    transform = transformations()

    image = transform(image).unsqueeze(0)

    outputs = model(image)
    mask = outputs["segmentation"].squeeze(0).detach().numpy()
    mask = mask.transpose(1, 2, 0)

    if classification:
        label = outputs["classification"].detach().numpy()[0]
        cl = label.argmax() + 1

    fig, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(image.squeeze(0).permute(1, 2, 0))
    ax[1].imshow(mask, cmap="gray")
    if classification:
        ax[1].set_title(f"Class: {label.argmax() + 1}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image", type=str, required=True)
    parser.add_argument("--path_to_model", type=str, required=True)
    args = parser.parse_args()

    path_to_image = args.path_to_image
    path_to_model = args.path_to_model

    image = Image.open(path_to_image).convert("RGB")

    num_classes = 9
    model = build_model(num_classes=num_classes, classification=False)
    model.load_state_dict(torch.load(path_to_model))

    print(model)

    visualize(image, model, classification=False)
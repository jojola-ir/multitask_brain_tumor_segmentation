import argparse
import os
from os.path import join

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataloader import MultitaskDataset, transformations
from loss import FocalLoss
from metrics import compute_hausdorff_monai, DiceScore
from model import build_model

ALPHA = 0.3
BETA = 0.7


def train(model, optimizer, train_loader, val_loader, epochs, num_classes, save_path, device):
    """Trains the model.
    Args:
    -----
        model: model, torch.nn.Module
        optimizer: optimizer, torch.optim
        criterion: loss function, torch.nn
        train_loader: train loader, torch.utils.data.DataLoader
        val_loader: validation loader, torch.utils.data.DataLoader
        epochs: number of epochs, int
        device: device, torch.device
    """
    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = FocalLoss()

    scheduler = ExponentialLR(optimizer, gamma=0.1)
    lrs = []

    loss_list = []

    # classification metrics
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    average_precision = torchmetrics.AveragePrecision(task="multiclass", num_classes=num_classes)
    acc_list = []
    ap_list = []

    # segmentation metrics
    dice = DiceScore()
    hausdorff = compute_hausdorff_monai()
    dice_list = []
    hausdorff_list = []

    for epoch in range(epochs):
        model.train()
        with tqdm(train_loader, unit="batch") as loader:
            loader.set_description(f"Train | Epoch {epoch + 1}/{epochs}")
            for batch in loader:
                loader.set_description(f"Epoch {epoch} / {epochs}")
                images, masks, labels = batch["image"], batch["mask"], batch["label"]
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)

                classification_loss = classification_criterion(outputs["classification"], labels)
                segmentation_loss = segmentation_criterion(outputs["segmentation"], masks)
                loss = ALPHA * classification_loss + BETA * segmentation_loss
                loss.backward()
                loss_list.append(loss.item())

                lrs.append(optimizer.param_groups[0]["lr"])
                optimizer.step()

                # classification metrics
                acc = accuracy(outputs["classification"], labels)
                ap = average_precision(outputs["classification"], labels)
                acc_list.append(acc.item())
                ap_list.append(ap.item())

                # segmentation metrics
                dice_score = dice(outputs["segmentation"], masks)
                hausdorff_score = hausdorff(outputs["segmentation"], masks)
                dice_list.append(dice_score.item())
                hausdorff_list.append(hausdorff_score.item())

                loader.set_postfix(loss=loss.item(), acc=acc.item(), ap=ap.item(), dice=dice_score.item(),
                                   hausdorff=hausdorff_score.item())

        val_loss_list = []
        val_acc_list = []
        val_ap_list = []
        val_dice_list = []
        val_hausdorff_list = []
        model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as loader:
                loader.set_description("Validation")
                for batch in loader:
                    images, masks, labels = batch["image"], batch["mask"], batch["label"]
                    images, masks, labels = images.to(device), masks.to(device), labels.to(device)

                    outputs = model(images)
                    classification_loss = classification_criterion(outputs["classification"], labels)
                    segmentation_loss = segmentation_criterion(outputs["segmentation"], masks)
                    loss = ALPHA * classification_loss + BETA * segmentation_loss
                    val_loss_list.append(loss.item())

                    # classification metrics
                    acc = accuracy(outputs["classification"], labels)
                    ap = average_precision(outputs["classification"], labels)
                    val_acc_list.append(acc.item())
                    val_ap_list.append(ap.item())

                    # segmentation metrics
                    dice_score = dice(outputs["segmentation"], masks)
                    hausdorff_score = hausdorff(outputs["segmentation"], masks)
                    val_dice_list.append(dice_score.item())
                    val_hausdorff_list.append(hausdorff_score.item())

                    loader.set_postfix(loss=loss.item(), acc=acc.item(), ap=ap.item(), dice=dice_score.item(),
                                       hausdorff=hausdorff_score.item())

        if epoch % 5 == 0:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), join(save_path, f"model_{epoch}.pth"))

        scheduler.step()

    torch.save(model.state_dict(), join(save_path, "model.pth"))

    fig_save = join(save_path, "figures")
    if not os.path.exists(fig_save):
        os.makedirs(fig_save)

    # Plot metrics
    fig, ax = plt.subplots()
    ax.plot(acc_list, color='red', label='Train accuracy')
    ax.plot(ap_list, linestyle='--', color='orange', label='Train average Precision')
    ax.plot(dice_list, linestyle='--', color='blue', label='Train dice Score')
    ax.plot(hausdorff_list, linestyle='--', color='green', label='Train hausdorff Distance')

    legend = ax.legend(loc='upper right', shadow=True)
    legend.get_frame().set_facecolor('#eafff5')
    plt.savefig(join(fig_save, f"train_metrics.png"))

    fig, ax = plt.subplots()
    ax.plot(val_acc_list, color='red', label='Val accuracy')
    ax.plot(val_ap_list, linestyle='--', color='orange', label='Val average Precision')
    ax.plot(val_dice_list, linestyle='--', color='blue', label='Val dice Score')
    ax.plot(val_hausdorff_list, linestyle='--', color='green', label='Val hausdorff Distance')

    legend = ax.legend(loc='upper right', shadow=True)
    legend.get_frame().set_facecolor('#eafff5')
    plt.savefig(join(fig_save, f"val_metrics.png"))

    # Plot loss
    fig, ax = plt.subplots()
    ax.plot(loss_list, color='red', label="Train loss")
    ax.plot(val_loss_list, color='red', label="Validation loss")

    legend = ax.legend(loc='upper right', shadow=True)
    legend.get_frame().set_facecolor('#eafff5')
    plt.savefig(join(fig_save, f"train_val_loss.png"))

    # Plot learning rate
    fig, ax = plt.subplots()
    ax.plot(lrs, label="Learning rate")

    legend = ax.legend(loc='upper right', shadow=True)
    legend.get_frame().set_facecolor('#eafff5')
    plt.savefig(join(fig_save, f"lr.png"))


def test(model, test_loader, device, num_classes):
    """Tests the model.
    Args:
    -----
        model: model, torch.nn.Module
        test_loader: test loader, torch.utils.data.DataLoader
        device: device, torch.device
    """
    model.eval()

    classification_criterion = nn.CrossEntropyLoss()
    segmentation_criterion = FocalLoss()

    loss_list = []

    # classification metrics
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
    average_precision = torchmetrics.AveragePrecision(task="multiclass", num_classes=num_classes)
    acc_list = []
    ap_list = []

    # segmentation metrics
    dice = DiceScore()
    hausdorff = compute_hausdorff_monai()
    dice_list = []
    hausdorff_list = []

    with torch.no_grad():
        with tqdm(test_loader, unit="batch") as loader:
            loader.set_description("Test")
            for batch in loader:
                images, masks, labels = batch["image"], batch["mask"], batch["label"]
                images, masks, labels = images.to(device), masks.to(device), labels.to(device)

                outputs = model(images)

                classification_loss = classification_criterion(outputs["classification"], labels)
                segmentation_loss = segmentation_criterion(outputs["segmentation"], masks)
                loss = ALPHA * classification_loss + BETA * segmentation_loss
                loss_list.append(loss.item())

                # classification metrics
                acc = accuracy(outputs["classification"], labels)
                ap = average_precision(outputs["classification"], labels)
                acc_list.append(acc.item())
                ap_list.append(ap.item())

                # segmentation metrics
                dice_score = dice(outputs["segmentation"], masks)
                hausdorff_score = hausdorff(outputs["segmentation"], masks)
                dice_list.append(dice_score.item())
                hausdorff_list.append(hausdorff_score.item())

                loader.set_postfix(loss=loss.item(), acc=acc.item(), ap=ap.item(), dice=dice_score.item(),
                                   hausdorff=hausdorff_score.item())


def main(path_to_data, batch_size, epochs, lr, save_path, device):
    """Main function.
    Args:
    -----
        path_to_data: path to data, str
        batch_size: batch size, int
        epochs: number of epochs, int
        lr: learning rate, float
        save_path: path to save model, str
        device: device, torch.device
    """
    # Load data
    print("Loading data...")
    dataset = MultitaskDataset(data=path_to_data, transform=transformations())

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    test_size = len(dataset) - train_size - val_size

    train_set, val_set = random_split(dataset, [train_size, val_size])
    train_set, test_set = random_split(train_set, [train_size, test_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4)

    for batch in train_loader:
        images, masks, labels = batch["image"], batch["mask"], batch["label"]
        break

    in_channels = images.shape[1]
    out_channels = masks.shape[1]
    num_classes = len(np.unique(labels))

    # Load model
    print("Creating model...")
    model = build_model(in_channels=in_channels, out_channels=out_channels, num_classes=num_classes)

    # Train model
    print("Training model...")
    train(model, train_loader, val_loader, epochs, lr, save_path, device)

    # Test model
    print("Testing model...")
    test(model, test_loader, device, num_classes=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--path_to_data", type=str, default="data")
    parser.add_argument("--batch_size", "-b", type=int, default=4)
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_path", type=str, default="results/")

    args = parser.parse_args()

    path_to_data = args.path_to_data
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    save_path = args.save_path

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    main(path_to_data, batch_size, epochs, lr, save_path, device)

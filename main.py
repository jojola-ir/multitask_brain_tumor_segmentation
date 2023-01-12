import argparse
import os
from os.path import join

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split

from torchvision import datasets

from skimage.metrics import peak_signal_noise_ratio as PSNR

from dataloader import build_data_pipes, get_noisy_image, transformations
from model import Unet


def evaluate(val_image, model, loss_fn, noise_parameter, device):
    model.eval()

    with torch.no_grad():
        data = get_noisy_image(val_image, noise_parameter)
        data = data.to(device)

        preds = model(data)

        target = torch.clone(val_image)
        target = target.to(device)

    psnr = PSNR(target.cpu().detach().numpy()[0], preds.cpu().detach().numpy()[0])
    mse = loss_fn(target.cpu().detach(), preds.cpu().detach())

    return psnr, mse


def train(train_image, val_image, model, optimizer, loss_fn, epochs, device, noise_parameter, results_path, color_space):
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    cs = "rgb" if color_space == 3 else "gray"

    data = get_noisy_image(train_image, noise_parameter)
    data = data.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs} | ", end="")
        for _ in range(10):
            preds = model(data)
            target = torch.clone(train_image)
            target = target.to(device)

            loss = loss_fn(preds, target)
            train_psnr = PSNR(target.cpu().detach().numpy()[0], preds.cpu().detach().numpy()[0])
            train_psnr = train_psnr.mean()

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        psnr, mse = evaluate(val_image, model, loss_fn, noise_parameter, device)
        print(f"train_psnr: {train_psnr:.3f} - train_mse: {loss:.5f}", end="")
        print(f" - val_psnr: {psnr:.3f} - val_mse: {mse:.5f}")

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_path, f"model_{cs}.pth"))

    torch.save(model.state_dict(), join(results_path, f"model_{cs}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images", "-p", type=str, default="../../../datasets/GAN/chest_xray/")
    parser.add_argument("--epochs", "-e", type=int, default=30)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-3)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--noise_parameter", "-n", type=float, default=0.2)
    parser.add_argument("--results", "-r", type=str, default="results/")
    parser.add_argument("--loader", action='store_true')

    args = parser.parse_args()

    dataset_path = args.path_to_images
    num_epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    noise_parameter = args.noise_parameter
    results_path = args.results
    loader = args.loader

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    transform = transformations()

    train_path = join(dataset_path, "train/")
    val_path = join(dataset_path, "val/")
    test_path = join(dataset_path, "test/")

    if loader:
        # TO_DO: correct DataPipe implementation
        # train_loader = build_data_pipes(train_path, transform, noise_parameter, batch_size)
        # val_loader = build_data_pipes(val_path, transform, noise_parameter, batch_size)
        # test_loader = build_data_pipes(test_path, transform, noise_parameter, batch_size)

        dataset = datasets.ImageFolder(dataset_path, transform=transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        inputs, _ = next(iter(train_loader))

    else:
        train_image = cv2.imread(join(train_path, "DSC00213.jpg"))
        val_image = cv2.imread(join(train_path, "DSC00213.jpg"))
        test_image = cv2.imread(join(train_path, "DSC00213.jpg"))

        train_image = transform(train_image).unsqueeze(0)
        val_image = transform(val_image).unsqueeze(0)
        test_image = transform(test_image).unsqueeze(0)

        print("Data loaded")

        inputs = train_image

    color_space = inputs.shape[1]

    model = Unet(in_channels=color_space, out_channels=color_space, depth=3).to(device=device)

    p = 0
    for param in model.parameters():
        p += param.numel()
    print(f"Number of parameters: {p}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train(train_image, val_image,
          model, optimizer, loss_fn,
          num_epochs, device, noise_parameter,
          results_path, color_space)

    psnr, mse = evaluate(test_image, model, loss_fn, noise_parameter, device)
    print(f"test_psnr : {psnr:.3f} - test_mse : {mse:.5f}")
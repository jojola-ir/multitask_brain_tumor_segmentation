import argparse
import os
from os.path import join

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_SIZE = 224


def transformations():
    """Applies transformations to an image.
    Args:
        image: image, np.array
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225]),
    ])
    return transform


class MultitaskDataset(Dataset):
    """Custom multitask dataset for classification and segmentation tasks."""

    def __init__(self, data, transform=None):
        super(MultitaskDataset, self).__init__()
        self.transform = transform

        dirs = os.listdir(data)
        extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
        self.data = []

        for dir in dirs:
            if not dir.startswith("."):
                subdir = join(data, dir)
                for file in os.listdir(join(subdir, dir)):
                    if file.endswith(extensions):
                        label = dirs.index(dir)
                        self.data.append({"image": join(join(subdir, dir), file),
                                          "mask": join(join(subdir, dir) + " GT", file),
                                          "label": label})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]

        image = Image.open(data["image"]).convert("RGB")
        mask = Image.open(data["mask"]).convert("L")
        label = data["label"]

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask, label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_images", "-p", type=str,
                        default="/Users/irina/Documents/Etudes/DS/datasets/Segmentation/large_scale_fish_dataset/Fish_Dataset/")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    args = parser.parse_args()

    dataset_path = args.path_to_images
    batch_size = args.batch_size

    dataset = MultitaskDataset(data=dataset_path, transform=transformations())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i, (image, mask, label) in enumerate(loader):
        print(image.shape, mask.shape, label)
        print(f"Image | mean: {image.mean():.3f}\t std: {image.std():.3f}\t shape: {image.shape}")
        print(f"Mask | mean: {mask.mean():.3f}\t std: {mask.std():.3f}\t shape: {mask.shape}")
        break

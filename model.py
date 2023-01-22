import torch
import torch.nn as nn

import torchvision.transforms.functional as F


class ConvBlock(nn.Module):
    """Convolutional block."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class HydraUnet(nn.Module):
    """Custom Unet model for segmentation and classification."""

    def __init__(self, in_channels=3, out_channels=1, in_features=32, depth=3, num_classes=2):
        super(HydraUnet, self).__init__()
        self.num_classes = num_classes
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # downstream
        for features in range(depth):
            feature = in_features * 2 ** features
            self.downs.append(ConvBlock(in_channels, feature))
            in_channels = feature

        # upstream
        for features in reversed(range(depth)):
            feature = in_features * 2 ** features
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature * 2, feature))

        # bottleneck
        self.bottleneck = ConvBlock(in_channels, in_channels * 2)

        # classifier
        self.classifier = nn.ModuleList()

        for features in reversed(range(depth)):
            feature = in_channels * 2 ** features // 2
            self.classifier.append(nn.Conv2d(feature, feature // 2, kernel_size=2, stride=2))
        self.classifier.append(nn.Flatten())

        self.conv_output = nn.Conv2d(in_features, out_channels, kernel_size=1)

    def forward(self, x):
        """Forward pass of the network."""
        outputs = {}

        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        x_cl = x
        for classifier in self.classifier:
            x_cl = classifier(x_cl)

        x_cl = nn.Linear(x_cl.shape[1], self.num_classes)(x_cl)
        if self.num_classes > 1:
            outputs['classification'] = nn.Softmax(dim=1)(x_cl)
        else:
            outputs['classification'] = nn.Sigmoid()(x_cl)

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]

            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])

            concat = torch.concat((skip, x), dim=1)
            x = self.ups[idx + 1](concat)

        x = self.conv_output(x)
        outputs['segmentation'] = x

        return outputs


def build_model(in_channels=3, out_channels=1, num_classes=2):
    """Builds the model.
    Args:
    -----
        num_classes: number of classes, int
    Returns:
    --------
        model: model, nn.Module
    """
    model = HydraUnet(in_channels=in_channels, out_channels=out_channels, num_classes=num_classes)
    return model


if __name__ == "__main__":
    model = build_model()

    print(model)

    p = 0
    for param in model.parameters():
        p += param.numel()

    print(f"Number of parameters: {p}")

    x = torch.randn(1, 3, 224, 224)

    outputs = model(x)
    print(outputs["classification"].shape)
    print(outputs["segmentation"].shape)

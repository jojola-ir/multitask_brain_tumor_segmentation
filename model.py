import torch
import torch.nn as nn
import torchvision.transforms.functional as F

class ConvBlock(nn.Module):
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


class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, in_features=32, depth=3):
        super(Unet, self).__init__()
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
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock(feature*2, feature))

        # bottleneck
        self.bottleneck = ConvBlock(in_channels, in_channels*2)

        self.output = nn.Conv2d(in_features, out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]

            if x.shape != skip.shape:
                x = F.resize(x, size=skip.shape[2:])

            concat = torch.concat((skip, x), dim=1)
            x = self.ups[idx+1](concat)

        return self.output(x)


class AutoEncoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, in_features=32, kernel_size=2, stride=2, depth=3):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for d in range(depth):
            features = in_features * 2 ** d
            self.encoder.append(nn.Conv2d(in_channels, features, kernel_size, padding="same"))
            self.encoder.append(nn.ReLU())
            #if d != depth - 1:
                #self.encoder.append(nn.MaxPool2d(kernel_size, stride))
            self.encoder.append(nn.MaxPool2d(kernel_size, stride))
            in_channels = features

        self.bottleneck = ConvBlock(in_channels, in_channels*2)

        for d in reversed(range(depth)):
            features = in_features * 2 ** d
            self.decoder.append(nn.ConvTranspose2d(features*2, features, kernel_size, stride))
            self.decoder.append(nn.ReLU())

        self.decoder.append(nn.ConvTranspose2d(features, out_channels, 1))
        self.decoder.append(nn.Sigmoid())

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)

        x = self.bottleneck(x)

        for layer in self.decoder:
            x = layer(x)
        return x


if __name__ == "__main__":
    model = Unet(in_channels=3, out_channels=3, depth=3)
    x = torch.randn((1, 3, 512, 512))
    print(model(x).shape)

    p = 0
    for param in model.parameters():
        p += param.numel()
    print(f"Number of parameters: {p}")
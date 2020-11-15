import torch
import torch.nn as nn
import torch.nn.functional as F

dependencies = ["torch"]


__all__ = ["AlexNet", "alexnet"]


class AlexNet(torch.nn.Module):
    def __init__(self, D_in=32, D_out=10):
        """
        D_in:
            Width of Image to apply network to
        D_out:
            Number of classes to output
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 5, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 384, 2)
        self.conv4 = nn.Conv2d(384, 384, 2)
        self.conv5 = nn.Conv2d(384, 256, 2)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, D_out)

    def forward(self, x):
        # Top 5 Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(F.relu(self.conv5(x)))

        # Flatten
        x = torch.flatten(x, start_dim=1)

        # FC endpoint
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def alexnet(pretrained=False, D_out=D_out, **kwargs):
    """# This docstring shows up in hub.help()
    AlexNet (7) model for 32x32 images
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = AlexNet(D_out=D_out)
    if pretrained:
        from pathlib import Path

        model_location = Path("./alexnet_cifar10-fine-tune.pth")
        state_dict = torch.load(model_location)
        model.load_state_dict(state_dict)
    return model

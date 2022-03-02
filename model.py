
import numpy as np
from torch import nn
import torchvision

from torch.nn import functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.net = efficientnet_pytorch.EfficientNet.from_name("efficientnet-b0")
        # checkpoint = torch.load("../input/efficientnet-pytorch/efficientnet-b0-08094119.pth")
        # self.net.load_state_dict(checkpoint)
        # n_features = self.net._fc.in_features
        # self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)
        self.map = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)
        self.net = torchvision.models.alexnet(num_classes=1)

        # self.net = torchvision.models.resnet50(num_classes=1)

    def forward(self, x):
        x = F.relu(self.map(x))

        out = self.net(x)
        return out


class LossMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        # incremental update
        self.avg = true_count / self.n + last_n / self.n * self.avg
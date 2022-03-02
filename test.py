import torchvision
from torch.nn import functional as F
from torch import nn

from torchsummary import summary

class CFG:
    img_size = 256
    n_frames = 16
    center_crop = 0

    cnn_features = 256
    lstm_hidden = 32

    n_fold = 5
    n_epochs = 30
    lr = 1e-4

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.map = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)

        self.net = torchvision.models.resnet18(num_classes=256)


    def forward(self, x):
        x = F.relu(self.map(x))
        out = self.net(x)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(CFG.cnn_features, CFG.lstm_hidden, 2, batch_first=True)
        self.fc = nn.Linear(CFG.lstm_hidden, 1, bias=True)

    def forward(self, x):
        # x shape: BxTxCxHxW
        batch_size, timesteps, C, H, W = x.size()
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, timesteps, -1)
        output, (hn, cn) = self.rnn(r_in)

        out = self.fc(hn[-1])
        return out

net = Model()
summary(net,(8,3,256,256))
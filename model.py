import torch as th
from torch import nn


class CoolModel(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate four parameters and assign them as
        member parameters.
        """
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.relu1 = nn.Tanh()
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=1, padding=2)
        self.relu2 = nn.Tanh()
        self.fc1 = nn.Linear(64 * 8 * 9, 64)
        self.relu3 = nn.Tanh()
        self.fc2 = nn.Linear(64, 64)
        self.relu4 = nn.Tanh()
        self.output = nn.Linear(64, 7)
        self.relu_out = nn.Tanh()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.relu_out(self.output(x))
        return x



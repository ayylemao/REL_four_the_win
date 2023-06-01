'''
Network class used as main logic of an agent playing connect 4.
'''

from torch import nn


class CoolModel(nn.Module):
    '''
    Network class used as main logic of an agent playing for connect 4.
    '''
    def __init__(self):
        """
        Definition of the network's layers.
        """
        super().__init__()       
        self.tanh = nn.Tanh()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=2, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=1, padding=2)
        self.fc1 = nn.Linear(11648, 364)
        self.fc2 = nn.Linear(364, 64)
        self.output = nn.Linear(64, 7)

    def forward(self, x):
        """
        Forward pass of the network.
        """
        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.tanh(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.output(x))
        return x

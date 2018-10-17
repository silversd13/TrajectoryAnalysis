# creates a class of neural network for trajectories analysis
import torch.nn as nn


class ConvNet(nn.Module):
    '''
    Network Architecture:
    Two Layers
    Layer 1 - Hidden Layer
        Op:  Conv      ->  ReLU      ->  MaxPool   ->  Dropout
        Sz: (1,2,1000) -> (1,16,990) -> (1,16,990) -> (1,16,494)
    Layer 2 - Hidden Layer
        Op: Conv       ->  ReLU      ->  MaxPool   ->  Dropout
        Sz: (1,16,494) -> (1,32,484) -> (1,32,484) -> (1,32,241)
    Layer 3 - Output Layer
        Op: Linear     -> ReLU
        Sz: (1,32,241) -> num_times
    '''

    def __init__(self):
        num_times = 5  # number of trial times to predict
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=.2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=2),
            nn.Dropout(p=.2))
        self.layer3 = nn.Sequential(
            nn.Linear(1*32*241, num_times),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        return out

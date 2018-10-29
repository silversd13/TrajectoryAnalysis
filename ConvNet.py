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

    def __init__(self, num_times=5, l1_conv_ks=11, l1_max_ks=4, l2_conv_ks=11, l2_max_ks=4):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=l1_conv_ks, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=l1_max_ks, stride=2),
            nn.Dropout(p=.2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=l2_conv_ks, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=l2_max_ks, stride=2),
            nn.Dropout(p=.2))
        self.layer3 = nn.Sequential(
            nn.Linear(int(1*32*(
                ((((((1000 - (l1_conv_ks-1)) - (l1_max_ks-1)-1)/2 + 1)
                    - (l2_conv_ks-1)) - (l2_max_ks-1)-1)/2 + 1))), num_times),
            nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        return out

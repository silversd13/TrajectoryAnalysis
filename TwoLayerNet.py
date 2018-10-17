# creates a class of neural network for trajectories analysis
import torch.nn as nn

class TwoLayerNet(nn.Module):
	'''
	Network Architecture:
	Two Layers
	Layer 1 - Linear -> ReLU # Hidden Layer
		Inp: 2 * 1000
		Out:  100
	Layer 2 - Linear -> ReLU # Output Layer
		Inp - 100
		Out - num_times (# time points during trial);
	'''
	def __init__(self):
		num_times = 5  # number of trial times to predict
		super(TwoLayerNet, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Linear(1000, 100),
			nn.ReLU())
		self.layer2 = nn.Sequential(
			nn.Linear(200, num_times),
			nn.ReLU())

	def forward(self, x):
		out = self.layer1(x)
		out = out.reshape(out.size(0), -1)
		out = self.layer2(out)
		return out
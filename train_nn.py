# script to train and save a neural network for the trajectories dataset

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, utils
from load_trajs import trajs_dataset
# from TwoLayerNet import TwoLayerNet
from ConvNet import ConvNet
from tensorboardX import SummaryWriter

# set up tensorbardx for data vis of loss
LOG_DIR = './runs'
writer = SummaryWriter(LOG_DIR + '/ConvNet_Adam_SL1_TotalLoss')
plot_iter = 0
loss_str = ['Move', 'Reach', 'Pellet', 'Grasp', 'Retract']
ind_writer = []
for j in range(5):
	ind_writer.append(SummaryWriter(LOG_DIR + '/ConvNet_Adam_SL1_' + loss_str[j]))

# create data loader
trajs = trajs_dataset(
	trial_times_file='labels.csv',
	trajs_dir='./CSV_files_DJT')
trajs_loader = torch.utils.data.DataLoader(trajs, batch_size=768, shuffle=True)

# instance of neural net
# nnet = TwoLayerNet()
nnet = ConvNet()
nnet.to('cpu')
nnet.train()

# set up loss function and optimizer
eta = .01
# criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(nnet.parameters(), lr=eta)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(nnet.parameters(), lr=eta)

# train neural net
num_epochs = 10000
for epoch in range(num_epochs):
	for i, sample in enumerate(trajs_loader):
		traj = sample['traj'].float().to('cpu')
		trial_times = sample['trial_times'].float().to('cpu')

		# Forward pass
		outputs = nnet(traj)
		loss = criterion(outputs, trial_times)
		# get individual losses too
		ind_loss = np.zeros((5,))
		for j in range(5):
			tmp = criterion(outputs[:, j], trial_times[:, j])
			ind_loss[j] = tmp.item()

		# Backward and optimize
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	# add graph to tensorboard
	if epoch == 0:
		writer.add_graph(model=nnet, input_to_model=traj)

	if (epoch % 5) ==0:
		# print progress and write to tensorboard log
		print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
			.format(epoch+1, num_epochs, i+1, len(trajs_loader), loss.item()))
		writer.add_scalar('Loss', loss.item(), plot_iter)
		for j in range(5):
			ind_writer[j].add_scalar('Loss', ind_loss[j], plot_iter)

		# add scatter plot to tensorboard
		fig = plt.figure(figsize=(10,7.5))
		for j in range(5):
			ax = fig.add_subplot(2, 3, j+1)
			ax.scatter(trial_times[:,j].detach().numpy(), outputs[:,j].detach().numpy(), s=1)
			ax.plot(ax.get_xlim(), ax.get_xlim(), '-k')
			ax.set_xlabel(loss_str[j] + 'Frames')
			ax.set_ylabel('Predicted ' + loss_str[j] + ' Frames')
		writer.add_figure('Predictions vs. True Times', fig, global_step=plot_iter)

		plot_iter += 1


# save model
torch.save(nnet.state_dict(), 'ConvNetState.ckpt')
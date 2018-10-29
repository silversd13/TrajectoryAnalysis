# script to train and save a neural network for the trajectories dataset

import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from load_trajs import trajs_dataset
from ConvNet import ConvNet
from tensorboardX import SummaryWriter

# try out different kernel sizes
conv_kernel_sizes = [3, 7, 11, 15, 21, 31, 41, 51]
max_kernel_sizes = [2, 4, 6, 8, 10]

for l1_conv_ks in conv_kernel_sizes:
	for l1_max_ks in max_kernel_sizes:
		for l2_conv_ks in conv_kernel_sizes:
			for l2_max_ks in max_kernel_sizes:
				# skip if conv net has odd dimensions
				dim = (((((((1000 - (l1_conv_ks - 1)) - (l1_max_ks - 1) - 1) / 2 + 1) - (l2_conv_ks - 1)) - (l2_max_ks - 1) - 1) / 2 + 1))
				if (dim != int(dim)):
					continue

				# set up tensorbardx for data vis of loss
				LOG_DIR = './runs/' + 'l1_conv_ks_' + str(l1_conv_ks) + '_l1_max_ks_' + str(l1_max_ks) + '_l2_conv_ks_' + str(l2_conv_ks) + '_l2_max_ks_' + str(l2_max_ks)
				print('\n\n' + 60*'*' + '\n' + LOG_DIR + '\n' + 60*'*' + '\n\n')
				writer = SummaryWriter(LOG_DIR + '/ConvNet_Adam_SL1_TotalLoss')
				plot_iter = 0

				# create data loader
				trajs = trajs_dataset(
					trial_times_file='labels.csv',
					trajs_dir='./CSV_files_DJT')
				trajs_loader = DataLoader(trajs, batch_size=768, shuffle=True)

				# instance of neural net
				nnet = ConvNet(num_times=5, l1_conv_ks=l1_conv_ks, l1_max_ks=l1_max_ks, l2_conv_ks=l2_conv_ks, l2_max_ks=l2_max_ks)
				nnet.to('cpu')
				nnet.train()

				# set up loss function and optimizer
				eta = .01
				criterion = nn.SmoothL1Loss()
				optimizer = torch.optim.Adam(nnet.parameters(), lr=eta)

				# train neural net
				num_epochs = 200
				for epoch in range(num_epochs):
					for i, sample in enumerate(trajs_loader):
						traj = sample['traj'].float().to('cpu')
						trial_times = sample['trial_times'].float().to('cpu')

						# Forward pass
						outputs = nnet(traj)
						loss = criterion(outputs, trial_times)

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

						plot_iter += 1

				# save model
				model_name = 'ConvNetState_' + 'l1_conv_ks_' + str(l1_conv_ks) + '_l1_max_ks_' + str(l1_max_ks) + '_l2_conv_ks_' + str(l2_conv_ks) + '_l2_max_ks_' + str(l2_max_ks)
				torch.save(nnet.state_dict(), model_name + '.ckpt')
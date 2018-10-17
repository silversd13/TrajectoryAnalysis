'''
Script to load data using the pytorch dataloader class
'''

# libs
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils

#
class trajs_dataset(torch.utils.data.Dataset):
    """trajectories dataset."""

    def __init__(self, trial_times_file, trajs_dir, transform=None):
        """
        Args:
            trial_times_file (string): Path to the trial times csv file.
            trajs_dir (string): Directory with all the trajectory csv files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.trial_times_frame = pd.read_csv(trial_times_file) # read in trial times @ init
        self.trajs_dir = trajs_dir
        self.transform = transform

    def __len__(self):
        return len(self.trial_times_frame)

    def __getitem__(self, idx):
        # get csv file name
        ratID = self.trial_times_frame.iloc[idx, 0]
        block = str(self.trial_times_frame.iloc[idx, 1])
        trial = str(self.trial_times_frame.iloc[idx, 2])
        filename = os.path.join(self.trajs_dir,ratID + '_vb' + block + '_tr' + trial + '.csv')
        # read in traj csv file
        traj_frame = pd.read_csv(filename)
        x = traj_frame['X Pix'].as_matrix()
        y = traj_frame['Y Pix'].as_matrix()
        traj = torch.from_numpy(np.vstack((x,y)))
        # get trial times for idx
        trial_times = self.trial_times_frame.iloc[idx, 3:]
        trial_times = torch.from_numpy(trial_times.as_matrix().astype(float))
        sample = {'traj': traj, 'trial_times': trial_times}

        if self.transform:
            sample = self.transform(sample)

        return sample
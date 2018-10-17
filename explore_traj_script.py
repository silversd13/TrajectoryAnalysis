'''
Script to explore dataset.
Trajectory data by animal in trajectories folder
'''

## import libs
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import time

## load the data
datadir = './trajectories'
ratID = 'T328'
block = 6
filename = datadir + '/' + ratID + '/' + 'video_block-' + str(block) + '.mat'
print('Loading File:', filename)

# keys = data.keys()
# 'trialFrameInd', 'trialFrames', 'pellet', 'trial','numFrames','newFrameTimes','trialNum','trial_new','wall'
data = io.loadmat(filename)
trial_new = np.squeeze(data['trial_new'])
newFrameTimes = np.squeeze(data['newFrameTimes'])
trialFrameInd = np.squeeze(data['trialFrameInd'])

trial = 20;
traj = trial_new[trial]['traj']
# paw = trial_new[trial]['pixel']
trial_time = newFrameTimes[trialFrameInd[trial] + np.arange(0,len(traj))];
trial_start = trial_time[0]
trial_time = trial_time - trial_start # relative to start of trial

# for frame in range(614):
# 	plt.cla()
# 	plt.show()
# 	plt.plot(traj[:,0],traj[:,1])
# 	plt.scatter(paw[frame][0][:,0],paw[frame][0][:,1])
# 	plt.show()
# 	time.sleep(.001)

# linear interpolation of trajectories
Fs = 50 # Hz
dt = 1/Fs
time = np.arange(0,trial_time[-1],step=dt)
x = np.interp(interp_trial_time,trial_time,traj[:,0])
y = np.interp(interp_trial_time,trial_time,traj[:,1])

# zeropad to 15s / if exceeds 15s cut
Nsamps = Fs*15
if len(time) > Nsamps:
	time = interp_trial_time[:Nsamps]
	x = x[:Nsamps]
	y = y[:Nsamps]
else:
	N = Nsamps - len(time)
	time = np.pad(time,(0,N),'constant',constant_values=(0,0))
	x = np.pad(x,(0,N),'constant',constant_values=(0,0))
	y = np.pad(y,(0,N),'constant',constant_values=(0,0))

# get times to predict
movStartFrame = trial_new[trial]['movStart'][-1][-1]
movStartTime = newFrameTimes[trialFrameInd[trial] + movStartFrame] - trial_start
movStartTime = interp_trial_time[np.abs(interp_trial_time - movStartTime).argmin()] # convert to interp trial times

reachFrame = trial_new[trial]['reach'][-1][-1]
reachTime = newFrameTimes[trialFrameInd[trial] + reachFrame] - trial_start
reachTime = interp_trial_time[np.abs(interp_trial_time - reachTime).argmin()] # convert to interp trial times

pelletFrame = trial_new[trial]['pellet'][-1][-1]
pelletTime = newFrameTimes[trialFrameInd[trial] + pelletFrame] - trial_start
pelletTime = interp_trial_time[np.abs(interp_trial_time - pelletTime).argmin()] # convert to interp trial times

graspFrame = trial_new[trial]['grasp'][-1][-1]
graspTime = newFrameTimes[trialFrameInd[trial] + graspFrame] - trial_start
graspTime = interp_trial_time[np.abs(interp_trial_time - graspTime).argmin()] # convert to interp trial times

retractFrame = trial_new[trial]['retract'][-1][-1]
retractTime = newFrameTimes[trialFrameInd[trial] + retractFrame] - trial_start
retractTime = interp_trial_time[np.abs(interp_trial_time - retractTime).argmin()] # convert to interp trial times


# save traj in npz
trial_ct = 1
sf = datadir + '/trial_%04d_trajectories' %trial_ct +  '.npz'
print('Saving Trajectories:', sf)
np.savez(sf,time=time, x=x, y=y)
# save times to predict in npz
sf = datadir + '/trial_%04d_times' %trial_ct +  '.npz'
print('Saving Times:', sf)
np.savez(sf,movStartTime=movStartTime,reachTime=reachTime,
	pelletTime=pelletTime,graspTime=graspTime,retractTime=retractTime)

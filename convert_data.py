'''
Script to convert/save all data.
Interpolates to get consistent sampling
Zeropads to get consistent sizes
'''

## import libs
import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import time

## load the data
datadir = './trajectories'
ratIDs = ['T254','T269','T276','T326','T328']
blocks = [[20,21,30,60,126],[6,9,32],[1,7,15],[10,17,26],[6,11,12,32,33]]

trial_ct = 0
for i,ratID in enumerate(ratIDs):
	for block in blocks[i]:
		filename = datadir + '/' + ratID + '/' + 'video_block-' + str(block) + '.mat'
		print('Loading File:', filename)

		data = io.loadmat(filename)
		trial_new = np.squeeze(data['trial_new'])
		newFrameTimes = np.squeeze(data['newFrameTimes'])
		trialFrameInd = np.squeeze(data['trialFrameInd'])

		for trial in range(len(trial_new)):
			try:
				trial_ct += 1
				traj = trial_new[trial]['traj']
				# paw = trial_new[trial]['pixel']
				trial_time = newFrameTimes[trialFrameInd[trial] + np.arange(0,len(traj))];
				trial_start = trial_time[0]
				trial_time = trial_time - trial_start # relative to start of trial

				# linear interpolation of trajectories
				Fs = 50 # Hz
				dt = 1/Fs
				time = np.arange(0,trial_time[-1],step=dt)
				x = np.interp(time,trial_time,traj[:,0])
				y = np.interp(time,trial_time,traj[:,1])

				# zeropad to 15s / if exceeds 15s cut
				Nsamps = Fs*15
				if len(time) > Nsamps:
					time = time[:Nsamps]
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
				movStartTime = time[np.abs(time - movStartTime).argmin()] # convert to interp trial times

				reachFrame = trial_new[trial]['reach'][-1][-1]
				reachTime = newFrameTimes[trialFrameInd[trial] + reachFrame] - trial_start
				reachTime = time[np.abs(time - reachTime).argmin()] # convert to interp trial times

				pelletFrame = trial_new[trial]['pellet'][-1][-1]
				pelletTime = newFrameTimes[trialFrameInd[trial] + pelletFrame] - trial_start
				pelletTime = time[np.abs(time - pelletTime).argmin()] # convert to interp trial times

				graspFrame = trial_new[trial]['grasp'][-1][-1]
				graspTime = newFrameTimes[trialFrameInd[trial] + graspFrame] - trial_start
				graspTime = time[np.abs(time - graspTime).argmin()] # convert to interp trial times

				retractFrame = trial_new[trial]['retract'][-1][-1]
				retractTime = newFrameTimes[trialFrameInd[trial] + retractFrame] - trial_start
				retractTime = time[np.abs(time - retractTime).argmin()] # convert to interp trial times


				# save traj in npz
				sf = datadir + 'Trial_Trajectories/trial_%04d' %trial_ct +  '.npz'
				print('Saving Trajectories:', sf)
				np.savez(sf,time=time, x=x, y=y)
				
				# save times to predict in npz
				sf = datadir + 'Trial_Times/trial_%04d' %trial_ct +  '.npz'
				print('Saving Times:', sf)
				np.savez(sf,ratID=ratID,trial=trial,
					movStartTime=movStartTime,reachTime=reachTime,
					pelletTime=pelletTime,graspTime=graspTime,retractTime=retractTime)
			except:
				print('skipped trial')
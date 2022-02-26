#!/usr/bin/env python

### libraries
import sys
import os
import pandas as pd
import mne
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from mne import io, read_proj, read_vectorview_selection
from mne.datasets import sample
from mne.time_frequency import psd_multitaper
import re 
from scipy import stats
from time import time
#%matplotlib inline
import warnings
warnings.filterwarnings('ignore')


### 1. LOAD DATA:

# Load iEEG file
subject = sys.argv[1] ################################################## CHANGE: SUBJECT NAME USED FOR EXPORT
pathEDF = sys.argv[2] ######################################### CHANGE: FILE PATH/NAME 
raw = mne.io.read_raw_edf(pathEDF, preload=True)
mne.set_log_level("WARNING")

# iEEG file info:
print('Data type: {}\n\n{}\n'.format(type(raw), raw))
print('Sample rate:', raw.info['sfreq'], 'Hz') # Get the sample rate
print('Size of the matrix: {}\n'.format(raw.get_data().shape)) # Get the size of the matrix
# print(raw.info) # VIEW INFO SUMMARY OF EEG DATA
# print('The actual data is just a matrix array!\n\n {}\n'.format(raw.get_data()))


### 2. CLEAN DATA:

def cleaner(raw):
    """
    iEEG PREPROCESSING PIPELINE
    INPUT: RAW iEEG (MNE)
    OUTPUT: CLEANED iEEG ('picks')
    # note: resampling should already be based on a filtered signal!
    # (i.e., first filtering, then down sampling)
    """
    ### 1. rereference data (average rereference)
    raw.set_eeg_reference('average', projection=True)
    # raw.plot_psd(area_mode='range', tmax=10.0) # visual verification
    print('Original sampling rate:', raw.info['sfreq'], 'Hz')

    ### 2. notch filter
    raw = raw.notch_filter(np.arange(60, int(raw.info['sfreq']/2)-1, 60), filter_length='auto', phase='zero') # 60, 241, 60
    # raw.plot_psd(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, n_fft=n_fft,
    #              n_jobs=1, proj=True, ax=ax, color=(1, 0, 0), picks=picks) # visual verification

    ### 3. other filters 
    # low pass filter (250Hz)
    raw = raw.filter(None, 250., h_trans_bandwidth='auto', filter_length='auto', phase='zero')
    # high pass filter (1Hz) - remove slow drifts
    raw = raw.filter(1., None, l_trans_bandwidth='auto', filter_length='auto', phase='zero')
    # raw.plot_psd(area_mode='range', tmax=10.0) # visual verification

    ### 4. downsampling (200Hz)
    raw = raw.resample(200, npad='auto')
    print('New sampling rate:', raw.info['sfreq'], 'Hz')
    
    ### 5. reject bad channels
    def check_bads_adaptive(raw, picks, fun=np.var, thresh=3, max_iter=np.inf):
        ch_x = fun(raw[picks, :][0], axis=-1)
        my_mask = np.zeros(len(ch_x), dtype=np.bool)
        i_iter = 0
        while i_iter < max_iter:
            ch_x = np.ma.masked_array(ch_x, my_mask)
            this_z = stats.zscore(ch_x)
            local_bad = np.abs(this_z) > thresh
            my_mask = np.max([my_mask, local_bad], 0)
            print('iteration %i : total bads: %i' % (i_iter, sum(my_mask)))        
            if not np.any(local_bad): 
                break
            i_iter += 1
        bad_chs = [raw.ch_names[i] for i in np.where(ch_x.mask)[0]]
        return (bad_chs)
    # Find the first index of the super-bad channels
    endIndex = 1
    for i, name in enumerate(raw.info['ch_names']): # can add new logic to reject other channels that are definitely bad
        if len(re.compile(r'C\d{3}').findall(name)) > 0:
            endIndex = i
            break
    bad_chs = raw.ch_names[endIndex:]
    bad_chs.extend(check_bads_adaptive(raw, list(range(0,endIndex)), thresh=3)) 
    raw.info['bads'] = bad_chs
    #     print(bad_chs)
    #     print(len(raw.info['bads'])) # check which channels are marked as bad
    ### PICK ONLY GOOD CHANNELS:
    picks = raw.pick_types(eeg = True, meg = False, exclude = 'bads')
    print("NUMBER OF CHANNELS FOR SUBJECT {}: {}".format(subject,len(picks.info['chs'])))
    #     print("THIS SHOULD BE 0: {}".format(len(picks.info['bads'])) ) # check statement
    
    return (picks)

picks = cleaner(raw)




### EXPORT DATA:
# export picks as .csv 
folderpathOutput=sys.argv[3]
if not os.path.isdir(folderpathOutput):
	os.mkdir(folderpathOutput)

header = ','.join(picks.ch_names)
np.savetxt(folderpathOutput+'/sub-'+subject+'_eegdata.csv', picks.get_data().T, delimiter=',', header=header) ### ---> spike detection




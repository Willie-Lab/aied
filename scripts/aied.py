
#!/usr/bin/env python3

# libraries
from __future__ import print_function, division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import h5py
import re
import shutil
import copy
import time
import random
import warnings 
import operator
from datetime import datetime
from tqdm import tqdm, tqdm_notebook
from matplotlib.pyplot import specgram
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchtext import data
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
import joblib
import csv
import warnings
warnings.filterwarnings('ignore')
#%matplotlib inline


### 1. LOAD DATA:

### Manual inputs (denoted by #-markers throughout):
subject = 'tester1' ###### CHANGE: subject name (make sure name does NOT have "_" in it; used for subject col and exportname)
eegfileloading = 'sample_eegdata.csv' ############################################################### CHANGE: filename here

### LOAD EEG DATA --- rows = channels, cols = timepoints
eegdir = os.getcwd() ################################################################################ CHANGE: dir here
# load file, checking for header
input_csv_file = eegdir+'/'+eegfileloading
with open(input_csv_file, 'rb') as csvfile:
    csv_test_bytes = csvfile.read(10)  # grab sample of .csv for format detection
    headertest = csv_test_bytes.decode("utf-8")
    if any(c.isalpha() for c in headertest) == True:
        data = pd.read_csv(input_csv_file, header=0)
        channels = data.columns
    else:
        data = pd.read_csv(input_csv_file, header=None)
        
### quick check: transpose if not in proper format (rows = chans, cols = timepoints) - build on this later.
if len(data) > len(data.columns):
    data = data.T
    if type(data[0][0]) == str:
        data = data.drop(data.columns[0], axis=1)
    data = data.astype(float)
    print('CHECK: Number of channels ~ %d' % len(data))
else:
    data = data.astype(float)

    
### AUTO DUMP IED IMAGES: clears dir containing spectrograms if produced in previous iteration
spectdir = eegdir+'/SPECTS/IEDS/' ################################################################## CHANGE: dir here
os.makedirs(spectdir, exist_ok = True)
for filename in os.listdir(spectdir):
    file_path = os.path.join(spectdir, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))



### 2. LOAD TEMPLATE-MATCHING DETECTOR FUNCTIONS:

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features."""
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)
    return ind

def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib is not available.')
    else:
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                    % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()
        
def locate_downsample_freq(sample_freq, min_freq=200, max_freq=340):
    min_up_factor = np.inf
    best_candidate_freq = None
    for candidate in range(min_freq, max_freq+1):
        down_samp_rate = sample_freq / float(candidate)
        down_factor, up_factor = down_samp_rate.as_integer_ratio()
        if up_factor <= min_up_factor:
            min_up_factor = up_factor
            best_candidate_freq = candidate
    return best_candidate_freq


def butter_bandpass(low_limit, high_limit, samp_freq, order=5):
    nyquist_limit = samp_freq / 2
    low_prop = low_limit / nyquist_limit
    high_prop = high_limit / nyquist_limit
    b, a = signal.butter(order, [low_prop, high_prop], btype='band')
    def bb_filter(data):
        return signal.filtfilt(b, a, data)
    return bb_filter


def detect(channel, samp_freq, return_eeg=False, temp_func=None, signal_func=None):
    # assume that eeg is [channels x samples]
    # Round samp_freq to the nearest integer if it is large
    if samp_freq > 100:
        samp_freq = int(np.round(samp_freq))
    down_samp_freq = locate_downsample_freq(samp_freq)
    template = signal.triang(np.round(down_samp_freq * 0.06))
    kernel = np.array([-2, -1, 1, 2]) / float(8)
    template = np.convolve(kernel, np.convolve(template, kernel, 'valid') ,'full')
    if temp_func:
        template = temp_func(template, samp_freq)
    if signal_func:
        channel = signal_func(channel, samp_freq)

    down_samp_rate = samp_freq / float(down_samp_freq)
    down_samp_factor, up_samp_factor = down_samp_rate.as_integer_ratio()
    channel = signal.detrend(channel, type='constant')
    results = template_match(channel, template, down_samp_freq)
    up_samp_results = [np.round(spikes * down_samp_factor / float(up_samp_factor)).astype(int) for spikes in results]
    if return_eeg:
        return up_samp_results, [channel[start:end] for start, end in results]
    else:
        return up_samp_results

def template_match(channel, template, down_samp_freq, thresh=7, min_spacing=0): #######@@@############################## CHANGE: d:7,0
    template_len = len(template)
    cross_corr = np.convolve(channel, template, 'valid')
    cross_corr_std = med_std(cross_corr, down_samp_freq)
    detections = []
    # catch empty channels
    if cross_corr_std > 0:
        # normalize the cross-correlation
        cross_corr_norm = ((cross_corr - np.mean(cross_corr)) / cross_corr_std)
        cross_corr_norm[1] = 0
        cross_corr_norm[-1] = 0
        # find regions with high cross-corr
        if np.any(abs(cross_corr_norm > thresh)):
            peaks = detect_peaks(abs(cross_corr_norm), mph=thresh, mpd=template_len)
            peaks += int(np.ceil(template_len / 2.)) # center detection on template
            peaks = [peak for peak in peaks if peak > template_len and peak <= len(channel)-template_len]
            if peaks:
                # find peaks that are at least (min_spacing) secs away
                distant_peaks = np.diff(peaks) > min_spacing * down_samp_freq
                # always keep the first peak
                to_keep = np.insert(distant_peaks, 0, True)
                peaks = [peaks[x] for x in range(len(peaks)) if to_keep[x] == True]
                detections = [(peak-template_len, peak+template_len) for peak in peaks]
    return np.array(detections)

def med_std(signal, window_len):
    window = np.zeros(window_len) + (1 / float(window_len))
    std = np.sqrt(np.median(np.convolve(np.square(signal), window, 'valid') - np.square(np.convolve(signal, window, 'valid'))))
    return std





### 3. RUN TEMPLATE-MATCHING DETECTOR:

def autoDetect(eegdata, samp_freq = 200, subject = subject):
    """
    AUTODETECT: DETECTS ALL SPIKES IN EACH CHANNEL
         INPUT: raw eeg file (preprocessed signal)
        OUTPUT: all_detections (list containing a list of arrays for all detections), 
                channel_names (eeg channel names corresponding to each detection list)
    """
    ### DETECT SPIKES:
    all_detections = []
    channel_names = []
    for i in range(eegdata.shape[0]):
        channel = eegdata.iloc[i,:].astype(float) # run on each row (chan)
        detections = detect(channel, samp_freq, return_eeg=False, temp_func=None, signal_func=None) 
        all_detections.append(detections)
        channel_names.append(int(float((eegdata.columns[i]))))

    ### REFORMAT SPIKES:
    detections = pd.DataFrame(all_detections)
    channels = pd.DataFrame(channel_names)
    spikes = pd.concat([channels,detections], axis = 1)
    newspikes = spikes.transpose() 
    newspikes.columns = newspikes.iloc[0]
    newspikes = newspikes.iloc[1:] # remove duplicate channel_name row 
    ### AUTO LONG-FORMATTING OF SPIKES
    spikeDf = pd.DataFrame() # empty df to store final spikes and spikeTimes 
    for idx, col in enumerate(newspikes.columns):
        # extract spikes for each column 
        tempSpikes = newspikes.iloc[:,idx].dropna() # column corresponding to channel with all spikes
        tempSpikes2 = tempSpikes.tolist() # convert series to list 
        # extract channel name for each spike (duplicate based on the number of spikes)
        tempName = tempSpikes.name # channel name 
        tempName2 = [tempName] * len(tempSpikes) # repeat col name by the number of spikes in this channel 
        tempDf = pd.DataFrame({'channel': tempName2, 'spikeTime': tempSpikes2})
        # save and append to final df 
        spikeDf = spikeDf.append(tempDf)
        spikeDf['fs'] = samp_freq
        spikeDf['subject'] = subject
    return(spikeDf)

spikes = autoDetect(data) ### eegfile, Fs, sessionname; kleen_fs=200, preprocess_fs=200

print("SPIKES DETECTED (TEMP MATCH) = ", len(spikes))
print("")
print(spikes[:3])




### 4. GENERATE INPUT IMAGES FOR CNN:

def spectimgs(eegdata, spikedf):    
    """
    SPECTS: GENERATE SPECTS FOR CNN
        INPUT: 1) eegdata, 2) spikedf (df from automated template-matching spike detector)
        OUTPUT: spects within ./SPECTS/IEDS
    """
    for i in tqdm(range(0,len(spikedf))): 
        samp_freq = int(float(spikedf.fs.values[0]))
        #######################################
        pad = 1 # d:1 number of seconds for window 
        dpi_setting = 300 # d:300
        Nfft = 128*(samp_freq/500) # d: 128 
        h = 3
        w = 3
        #######################################
        try:
            subject = spikedf.subject.values[0]
            chan_name = int(spikedf.channel.values[i]) # zero idxed -1
            spikestart = spikedf.spikeTime.values[i][0] # start spike
            ### select eeg data row 
            ecogclip = eegdata.iloc[chan_name]
            ### filter out line noise
            b_notch, a_notch = signal.iirnotch(60.0, 30.0, samp_freq)
            ecogclip = pd.Series(signal.filtfilt(b_notch, a_notch, ecogclip)) 
        
            ### trim eeg clip based on cushion            
            ### mean imputation if missing indices
            end = int(float((spikestart+int(float(pad*samp_freq)))))
            start = int(float((spikestart-int(float(pad*samp_freq)))))
            if end > max(ecogclip.index):
                temp = list(ecogclip[list(range(spikestart-int(float(pad*samp_freq)), max(ecogclip.index)))])
                cushend = [np.mean(ecogclip)]*(end - max(ecogclip.index))
                temp = np.array(temp + cushend)
            elif start < min(ecogclip.index):
                temp = list(ecogclip[list(range(min(ecogclip.index), spikestart+pad*samp_freq))])
                cushstart = [np.mean(ecogclip)]*(min(ecogclip.index)-start)
                temp = np.array(cushstart, temp)
            else:
                temp = np.array(ecogclip[list(range(spikestart-int(float(pad*samp_freq)), 
                                         spikestart+int(float(pad*samp_freq))))]) 
            
            ### PLOT AND EXPORT:
            plt.figure(figsize=(h,w))
            specgram(temp, NFFT = int(Nfft), Fs = samp_freq, noverlap=int(Nfft/2), detrend = "linear", cmap = "YlOrRd") 
            plt.axis("off")
            plt.xlim(0, pad*2)
            plt.ylim(0,100)
            plt.savefig(spectdir+subject+"_"+str(spikestart)+"_"+str(chan_name)+".png", dpi = dpi_setting)
            plt.close()
        except Exception as e: 
            print(e)
            print("ERROR with IED portion:", i)
            plt.close()
            continue

spectimgs(data, spikes)




### 5. ResNet-18 CNN DETECTOR:

### A: LOAD ALL DATA --- extract clip_id from path
model_dir = eegdir+"/" # dir with trained model ################################################## CHANGE: dir here
proj_dir = eegdir+"/" # dir with main project script ############################################# CHANGE: dir here
imgs = 'SPECTS' # dir with IED / NONIED image dirs (name of subdir)

data_transforms = {
    imgs: transforms.Compose([
        transforms.Resize(224),
        transforms.Pad(1, fill=0, padding_mode='constant'),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder
    """
    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return (tuple_with_path)
    
image_datasets = {x: ImageFolderWithPaths(os.path.join(proj_dir, x),
                                          data_transforms[x]) for x in [imgs]}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1, # use batch=1, shuffle=F
                                             shuffle=False, num_workers=0) for x in [imgs]} 
class_names = image_datasets[imgs].classes
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# extract image paths
path_names = []
for images,labels,paths in dataloaders[imgs]:
    path_names.append(paths)
# convert list of paths to dataframe col
df = pd.DataFrame(path_names)
df.columns = ['clip_ids']
df[['clip_ids','clip']] = df['clip_ids'].str.split('IEDS/',expand=True)
df['clip'] = df['clip'].str.rstrip('.png')
df[['subject','start','chan']] = df['clip'].str.split('_',expand=True)

##############################################

### B: LOAD PRETRAINED MODEL 
try:
    model = torch.load(model_dir+'model_aied.pt')
    # model.eval() # model architecture
except ImportError:
    print('TRAINED MODEL NOT FOUND: Check that trained model is in eegdir and name matches: model_aied.pt')
    
###############################################

### C: RUN MODEL
y_pred = []
with torch.no_grad():
    for inputs,labels,paths in dataloaders[imgs]:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model.forward(inputs)
        _,predicted = torch.max(outputs, 1) 
        pred = predicted.numpy()
        lab = labels.numpy()
        y_pred.append(pred)

# reformat outputs:
y_pred_flat = np.concatenate((y_pred),axis=0)
# classes = ['class 0', 'class 1']
df['predicted_class'] = y_pred_flat
# # export as .csv
# df.to_csv(proj_dir+'all_predictions.csv', encoding='utf-8', index=False)
# print(df[:3]) # here, 1 = nonied, 0 = ied (df)




# 6. CLEAN SPIKE DF FOR EXPORT:

def dataCleaner(df, samp_freq = spikes.fs.values[0], win = 3): 
    """
    CLEANS SPIKE DATA FOR EXPORT
    INPUT: df from resnet model, win = number of seconds allowed for spike overlap 
                                (i.e., spikes within 3s of ea.other = single event))
    OUTPUT: clean df with subjectid, spikeStart, 
    channels (where spikes detected) - if contactName present, change to rownames, 
    numChannels (# channels spike detected)
    """
    ### only keep spikes: predicted_class = 0
    df = df[df.predicted_class == 0]
    df['start'] = df['start'].astype(int) # convert from str to int
    ### sort start times in df:
    df = df.sort_values(by = 'start', ascending = True)
    ### dedupe spikes by col and time
    bins =  np.arange(min(df.start.values), max(df.start.values), samp_freq*win)
    spikebins = np.digitize(df['start'], bins)
    cleandf = df.groupby(spikebins)['start'].describe()
    chanlist = df.groupby(spikebins)['chan'].apply(lambda x: x.values.tolist())
    chanlist = [list(set(x)) for x in chanlist]
    chancounts = [len(l) for l in chanlist]
    meanspikestart = (cleandf['mean']).astype(int)
    subjectid = [subject]*len(meanspikestart)
    ### reformat into new df
    finaldf = pd.DataFrame({'subject': subjectid, 'spikeStart': meanspikestart, 
                            'channels': chanlist, 'numChannels': chancounts})
    ### reject spikes detected in >= 12 channels within time window
    finaldf = finaldf[finaldf.numChannels < 12]
    return (finaldf)

finaldf = dataCleaner(df)

### export as .csv
finaldf.to_csv(proj_dir+subject+'_finalspikes.csv', encoding='utf-8', index=False) ############################ CHANGE: export name
print(finaldf[:3])
print("")
print("FINAL SPIKES DETECTED = ", len(finaldf))





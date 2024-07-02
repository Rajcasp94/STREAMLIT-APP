import streamlit as st
import pandas as pd
import numpy as np

#import neurokit2 as nk
import scipy
#from scipy.signal import find_peaks, medfilt
#import neurokit2 as nk
from statistics import mode
#import scipy.signal as signal
from scipy import stats


ef normalize(dataframe):
    dataframe = (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min()) 
    return dataframe
def butterworth(raw_signal,n,desired_cutoff,sample_rate,btype):
    if(btype=='high' or btype=='low'):
        critical_frequency=(2*desired_cutoff)/sample_rate
        B, A = scipy.signal.butter(n, critical_frequency, btype=btype, output='ba')
    elif(btype=='bandpass'):
        critical_frequency_1=(2*desired_cutoff[0])/sample_rate
        critical_frequency_2=(2*desired_cutoff[1])/sample_rate
        B, A = scipy.signal.butter(n, [critical_frequency_1,critical_frequency_2], btype=btype, output='ba')
    filtered_signal = scipy.signal.filtfilt(B,A, raw_signal)
    return filtered_signal
    
def extract_pat(ecg_signal, ppg_signal, ecg_time, ppg_time):
    pat = []
    #signals, info = nk.ecg_peaks(ecg_signal, sampling_rate = 500)
    ecg_peaks, _ = scipy.signal.find_peaks(ecg_signal, distance = 200)
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal, distance = 200)
    #st.write('length of the peak is',ecg_peaks)
    #ecg_peaks = list(info.values())[0]
    ecg_peak_times = [ecg_time[e] for e in ecg_peaks]
    ppg_peak_times = [ppg_time[p] for p in ppg_peaks]
    
    parse_length = min(len(ecg_peaks),len(ppg_peaks))
    i = 0
    j = 0
    while(i < parse_length-1 and j < parse_length-1):
        current_ecg = ecg_peak_times[i]
        next_ecg = ecg_peak_times[i + 1]
        current_ppg = ppg_peak_times[j]
        if((current_ppg > current_ecg) and (current_ppg <= next_ecg)):
            pat.append((current_ppg - current_ecg) * 0.002)
            i = i + 1
            j = j + 1
        if current_ppg > next_ecg:
            i = i + 1
            j = j
        if current_ppg <= current_ecg:
            i = i
            j = j + 1
    df_pat_list = pd.DataFrame({
        'PAT':pat
    })
    median_pat_list_filtered_df = df_pat_list.rolling(window=int(len(df_pat_list)/2)).median()
    median_pat_list_filtered_df = median_pat_list_filtered_df.fillna(np.mean(median_pat_list_filtered_df))
    median_pat_list_filtered = list(median_pat_list_filtered_df.values.flatten())
    return median_pat_list_filtered

st.title('Get your PAT')

sample_rate = st.number_input('Sampling rate (Hz)')


if sample_rate!= 0: 
    st.write('sample rate is', sample_rate)
else:
    st.write('Enter Sample Rate')

 
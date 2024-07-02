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

st.title('Get your PAT')

sample_rate = st.number_input('Sampling rate (Hz)')


if sample_rate!= 0: 
    st.write('sample rate is', sample_rate)
else:
    st.write('Enter Sample Rate')

 
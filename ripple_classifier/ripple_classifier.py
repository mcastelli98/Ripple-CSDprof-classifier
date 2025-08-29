"""
Functions used for ripple type classification.
Created on 29th August 2025

Author: Manfredi Castelli - manfredi.castelli98@gmail.com
"""
import numpy as np

from ripple_classifier import utils
from scipy import stats
from seaborn import color_palette

## Load model that can estimate LM CSD from ca1 LFP
filename = './models/ripple_type_lfp_classifier.lda'

model_saved = np.load(filename,allow_pickle=True)
model = model_saved['model']
lda_transformer = model.named_steps['lineardiscriminantanalysis']
model_info = model_saved['info']

# Average CA1 waveform used to train the model which can be used for channel selection
refLFP1 = model_info['refLFP']

winLen_classifer_ms = 100 
####### FUNCTIONS ############################
def model_report():
    print('#### REPORT ON TRAINED MODEL ####')
    print('Dataset used to train the LDA model contained {} ripples from 5 mice. ' \
          .format(model_info['n_swrs']))
######################################################
def format_lfp_inputs_to_classifier(pyr_lfp,taxis,winLen_classifer_ms=winLen_classifer_ms,sr=1250.):
    # Reformat classifier input to match number of features (i.e. time window)
    taxis_mask_model_input = np.digitize(taxis,[-winLen_classifer_ms,winLen_classifer_ms])==1
    taxis_classifier_ms = taxis[taxis_mask_model_input]
    # Reformat LFP data into (nevents, 250)
    return pyr_lfp[:,taxis_mask_model_input],taxis_classifier_ms

    
def norm_traces(lfp_trig):
    return np.nan_to_num([stats.zscore(lfp, nan_policy='omit') for lfp in lfp_trig])


def get_classes_idxs(y):
    '''Estimates the index of lm_sink, intermediate, rad_sink ripples
        returns lm_sink, intermediate, rad_sink'''
    return [np.where(y==i)[0] for i in range(3)]

def predict_ripple_types(lfp_rip,cut_off_Hz=30):
    '''Predict the ripple type using the LDA trained from the LFP of the CA1 pyramidal layer'''
    # Pre-process LFP Traces
    # 1) Normalise traces
    lfp_rip_norm = norm_traces(lfp_rip)
    # 2) Low-pass filter to extract sharp-wave component
    lfp_rip_norm = np.array([utils.lowpass(lfp,cut_off_Hz) for lfp in lfp_rip_norm])

    # Predict Ripple type
    ripple_lbls = model.predict(lfp_rip_norm)
    lm_sink, intermediate, rad_sink = get_classes_idxs(ripple_lbls)
    return lm_sink, intermediate, rad_sink
import os
import numpy as np
import pandas as pd
import logging
from sklearn import model_selection
from scipy import signal

import lightgbm as lgb
#----------------------------》
import numpy as np
import os, json
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from scipy import signal  # 
#from SelfSupervised_Trainer import *
#from models import EEGM2, EEGM2_S1, EEGM2_S3, EEGM2_S4, EEGM2_S5, UNet
#import TUABLoader
from Dataset.TUABLoader import TUABLoader

import pickle
import torch
#import matplotlib.pyplot as plt
#import torchvision
from collections import Counter
#----------------------------------《

logger = logging.getLogger(__name__)


def load(config):
    # Build data
    Data = {}
    if os.path.exists(config['data_dir'] + '/' + config['problem'] + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + config['problem'] + '.npy', allow_pickle=True)

        if np.any(Data_npy.item().get('val_data')):
            Data['train_data'] = Data_npy.item().get('train_data')
            Data['train_label'] = Data_npy.item().get('train_label')
            Data['val_data'] = Data_npy.item().get('val_data')
            Data['val_label'] = Data_npy.item().get('val_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[1]
            Data['All_train_data'] = Data_npy.item().get('All_train_data')
            Data['All_train_label'] = Data_npy.item().get('All_train_label')
            if config['Pre_Training'] == 'Cross-domain':
                Data['pre_train_data'], Data['pre_train_label'] = Cross_Domain_loader(Data_npy)
                logger.info(
                    "{} samples will be used for self-supervised Pre_training".format(len(Data['pre_train_label'])))
        else:
            Data['train_data'], Data['train_label'], Data['val_data'], Data['val_label'] = \
                split_dataset(Data_npy.item().get('train_data'), Data_npy.item().get('train_label'), 0.1)
            Data['All_train_data'] = Data_npy.item().get('train_data')
            Data['All_train_label'] = Data_npy.item().get('train_label')
            Data['test_data'] = Data_npy.item().get('test_data')
            Data['test_label'] = Data_npy.item().get('test_label')
            Data['max_len'] = Data['train_data'].shape[2]
    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))

    return Data


def Cross_Domain_loader(domain_data):
    All_train_data = domain_data.item().get('All_train_data')
    All_train_label = domain_data.item().get('All_train_label')
    # Load DREAMER for Pre-Training
    DREAMER = np.load('Dataset/DREAMER/DREAMER.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, DREAMER.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, DREAMER.item().get('All_train_label')), axis=0)

    # Load Crowdsource for Pre-Training
    Crowdsource = np.load('Dataset/Crowdsource/Crowdsource.npy', allow_pickle=True)
    All_train_data = np.concatenate((All_train_data, Crowdsource.item().get('All_train_data')), axis=0)
    All_train_label = np.concatenate((All_train_label, Crowdsource.item().get('All_train_label')), axis=0)
    return All_train_data, All_train_label

def tuev_loader(config):
    Data = {}
    data_path = config['data_dir'] + '/' + config['problem']
    Data['train_data'] = np.load(data_path + '/' + 'train_data.npy', allow_pickle=True)
    Data['train_label'] = np.load(data_path + '/' + 'train_label.npy', allow_pickle=True)
    Data['val_data'] = np.load(data_path + '/' + 'val_data.npy', allow_pickle=True)
    Data['val_label'] = np.load(data_path + '/' + 'val_label.npy', allow_pickle=True)
    Data['All_train_data'] = np.load(data_path + '/' + 'All_train_data.npy', allow_pickle=True)
    Data['All_train_label'] =np.load(data_path + '/' + 'All_train_label.npy', allow_pickle=True)
    Data['test_data'] = np.load(data_path + '/' + 'test_data.npy', allow_pickle=True)
    Data['test_label'] = np.load(data_path + '/' + 'test_label.npy', allow_pickle=True)
    Data['max_len'] = Data['train_data'].shape[1]

    logger.info("{} samples will be used for self-supervised training".format(len(Data['All_train_label'])))
    logger.info("{} samples will be used for fine tuning ".format(len(Data['train_label'])))
    samples, channels, time_steps = Data['train_data'].shape
    logger.info(
        "Train Data Shape is #{} samples, {} channels, {} time steps ".format(samples, channels, time_steps))
    logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
    logger.info("{} samples will be used for test".format(len(Data['test_label'])))
    return Data


    
def TUEV2_loader(config):

    
    root_path=r"/public/home/linbingru2024/dataset/TUEV/processed"
    #
    # 
    seed = 4523
    np.random.seed(seed)

    # 
    train_files = os.listdir(os.path.join(root_path, "processed_train"))
    val_files = os.listdir(os.path.join(root_path, "processed_eval"))
    test_files = os.listdir(os.path.join(root_path, "processed_test"))
    
    
    
    # 
    
    train_data, train_labels = load_data_from_files(
        os.path.join(root_path, "processed_train"), train_files
    )
    
    
    val_data, val_labels = load_data_from_files(
        os.path.join(root_path, "processed_eval"), val_files
    )
    
    
    test_data, test_labels = load_data_from_files(
        os.path.join(root_path, "processed_test"), test_files
    )
    
    # 
    all_train_data = np.concatenate([train_data, val_data], axis=0)
    all_train_labels = np.concatenate([train_labels, val_labels], axis=0)
    
    # 
    max_len = len(train_data)+len(val_data)
    
    # 
    Data = {
        'train_data': train_data,
        'train_label': train_labels,
        'val_data': val_data,
        'val_label': val_labels,
        'test_data': test_data,
        'test_label': test_labels,
        'max_len': max_len,
        'All_train_label': all_train_labels,
        'All_train_data': all_train_data
    }
    

    
    return Data


def tuab_loader(config):

    tuab_root=r"/public/home/liulonglong2024/lon/dataset/processed_256hz_2000seqlen_JH"
    train_files = os.listdir(os.path.join(tuab_root, "train"))
    val_files = os.listdir(os.path.join(tuab_root, "val"))
    test_files = os.listdir(os.path.join(tuab_root, "test"))
    # Create dataset objects
    train_data = TUABLoader((os.path.join(tuab_root, "train"), train_files))
    val_data = TUABLoader((os.path.join(tuab_root, "val"), val_files))
    test_data = TUABLoader((os.path.join(tuab_root, "test"), test_files))
    

    
    # 
    data = {}
    
    # 
    train_samples = len(train_data)
    train_data_array = []
    train_label_array = []
    
    features = []
    labels = []
    for i in range(len(train_data)):
        sample, label = train_data[i]
        sample_np = sample.numpy()
        mean = sample_np.mean(axis=1)
        std = sample_np.std(axis=1)
        feat = np.concatenate([mean, std])
        features.append(feat)
        labels.append(label)
    features = np.array(features)
    labels = np.array(labels)
    
    top_indices = select_top_channels_by_importance_from_features(features, labels, top_k=16)
    
    for i in range(train_samples):
        sample, label = train_data[i]
        sample_np = sample.numpy()
        sample_np = sample_np[top_indices, :]
        # 
        resampled_data = np.zeros((sample_np.shape[0], 350)) #256
        for ch in range(sample_np.shape[0]):
            resampled_data[ch] = signal.resample(sample_np[ch], 350) #256
        train_data_array.append(resampled_data)
        train_label_array.append(label)
    
    data['train_data'] = np.array(train_data_array)
    data['train_label'] = np.array(train_label_array)
    
    # 
    val_samples = len(val_data)
    val_data_array = []
    val_label_array = []
    
    for i in range(val_samples):
        sample, label = val_data[i]
        sample_np = sample.numpy()
        sample_np = sample_np[top_indices, :]
        # 
        resampled_data = np.zeros((sample_np.shape[0], 350)) #256
        for ch in range(sample_np.shape[0]):
            resampled_data[ch] = signal.resample(sample_np[ch], 350) #256
        val_data_array.append(resampled_data)
        val_label_array.append(label)
    
    data['val_data'] = np.array(val_data_array)
    data['val_label'] = np.array(val_label_array)
    
    # 
    test_samples = len(test_data)
    test_data_array = []
    test_label_array = []
    
    for i in range(test_samples):
        sample, label = test_data[i]
        sample_np = sample.numpy()
        sample_np = sample_np[top_indices, :]
        #
        resampled_data = np.zeros((sample_np.shape[0], 350)) #256
        for ch in range(sample_np.shape[0]):
            resampled_data[ch] = signal.resample(sample_np[ch], 350) #256
        test_data_array.append(resampled_data)
        test_label_array.append(label)
    
    data['test_data'] = np.array(test_data_array)
    data['test_label'] = np.array(test_label_array)
    
    # 
    data['All_train_data'] = data['train_data']
    data['All_train_label'] = data['train_label']
    data['max_len'] = len(data['All_train_data'])
    

    Data = data
    return Data


def PhysioNetP300_loader(config):
    # 
    ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7',
                'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz',
                'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz',
                'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4',
                'O2']
    ch_names = [x.upper() for x in ch_names]

    # 
    use_channels_names1 = ['FP1', 'FPZ', 'FP2',
                           'AF3', 'AF4',
                           'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
                           'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
                           'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
                           'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
                           'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                           'PO7', 'PO3', 'POZ', 'PO4', 'PO8',
                           'O1', 'OZ', 'O2', ]

    # 
    use_channels_names = []
    channels_index = []
    for x in use_channels_names1:
        if x in ch_names:
            channels_index.append(ch_names.index(x))
            use_channels_names.append(x)
    Data = {}

    # 

    data_dict = load_physio_p300_data()

    # 
    data_0 = data_dict.get("data_0", torch.tensor([]))
    data_1 = data_dict.get("data_1", torch.tensor([]))
    subjects_0 = data_dict.get("subjects_0", [])
    subjects_1 = data_dict.get("subjects_1", [])

    # 
    data_0_np = data_0.numpy() if data_0.numel() > 0 else np.array([])
    data_1_np = data_1.numpy() if data_1.numel() > 0 else np.array([])

    # 
    labels_0 = np.zeros(len(data_0_np), dtype=int)
    labels_1 = np.ones(len(data_1_np), dtype=int)

    # 
    all_data = np.concatenate([data_0_np, data_1_np], axis=0) if data_0_np.size > 0 and data_1_np.size > 0 else \
        data_1_np if data_1_np.size > 0 else data_0_np
    all_labels = np.concatenate([labels_0, labels_1]) if labels_0.size > 0 and labels_1.size > 0 else \
        labels_1 if labels_1.size > 0 else labels_0


    total_samples = len(all_data)
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)

    # 
    indices = np.random.permutation(total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 
    Data['train_data'] = all_data[train_indices].astype(np.float32)
    Data['train_label'] = all_labels[train_indices]
    Data['val_data'] = all_data[val_indices].astype(np.float32)
    Data['val_label'] = all_labels[val_indices]
    Data['test_data'] = all_data[test_indices].astype(np.float32)
    Data['test_label'] = all_labels[test_indices]


    if all_data.size > 0:
        num_channels = all_data.shape[1]
    else:
        num_channels = len(channels_index)

    Data['max_len'] = num_channels  # 


    train_val_indices = np.concatenate([train_indices, val_indices])
    Data['All_train_data'] = all_data[train_val_indices].astype(np.float32)
    Data['All_train_label'] = all_labels[train_val_indices]



    return Data


def fine_tune_data(Data, label, samples_per_class):
    # Randomly select samples from each class
    selected_indices = []
    for class_label in np.unique(label):
        indices = np.where(label == class_label)[0]
        selected_indices.extend(np.random.choice(indices, size=samples_per_class))
    # Select the corresponding data and labels
    selected_data = Data[selected_indices]
    selected_labels = label[selected_indices]

    return selected_data, selected_labels


def split_dataset(data, label, validation_ratio):
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label


def load_Preprocessed_DD(file_path, norm=True):
    Data = {}
    all_data = pd.read_csv(file_path)
    # Clean noisy recording = 'X', 'EyesCLOSEDneutral', 'EyesOPENneutral', 'LateBoredomLap'
    all_data = clean_Preprocessed_DD(all_data)
    all_labels = all_data.label__desc  # separate label
    # ------------------------------------------------------------------------------
    # Remove extra information -----------------------------------------------------
    subject_index = all_data.subject_id__desc
    col_index = find_feat_col(all_data)  # Remove description
    value = all_data[col_index]  # Remove description
    # -----------------------------------------------------------------------------
    # Split data to Train, Validation and Test -----------------------------------

    subject_uniq = subject_index.unique()
    test_subjects = subject_uniq[
        np.random.choice(len(subject_uniq), 3, replace=False)]  # Randomly select 3 subjects for testing
    valid_subject = np.random.choice(subject_uniq, 1)  # Randomly select 1 subject for validation

    test_subject_index = np.isin(subject_index, test_subjects)
    valid_subject_index = np.array(subject_index == valid_subject[0])

    test_data = value[test_subject_index]
    test_label = all_labels[test_subject_index]

    valid_data = value[valid_subject_index]
    valid_label = all_labels[valid_subject_index]

    # The remaining subjects will be used for validation
    train_subject_index = ~(test_subject_index | valid_subject_index)
    train_data = value[train_subject_index]
    train_label = all_labels[train_subject_index]

    test_data = reshape_DD(test_data)
    valid_data = reshape_DD(valid_data)
    train_data = reshape_DD(train_data)
    '''
    test_data = preprocess_eeg_data(reshape_DD(test_data), high_freq=50, low_freq=0.1, fs=128)
    valid_data = preprocess_eeg_data(reshape_DD(valid_data), high_freq=50, low_freq=0.1, fs=128)
    train_data = preprocess_eeg_data(reshape_DD(train_data), high_freq=50, low_freq=0.1, fs=128)
    '''


    logger.info("{} samples will be used for training".format(len(train_label)))
    logger.info("{} samples will be used for validation".format(len(valid_label)))
    logger.info("{} samples will be used for testing".format(len(test_label)))

    Data['max_len'] = train_data.shape[-1]
    Data['All_train_data'] = np.concatenate([train_data, valid_data])
    Data['All_train_label'] = np.concatenate([train_label.to_numpy(), valid_label.to_numpy()])
    Data['train_data'] = train_data
    Data['train_label'] = train_label.to_numpy()
    Data['val_data'] = valid_data
    Data['val_label'] = valid_label.to_numpy()
    Data['test_data'] = test_data
    Data['test_label'] = test_label.to_numpy()
    return Data


def load_Crowdsource(file_path):
    Data = {}
    all_data = pd.read_csv(file_path)
    label_type = all_data.labels_i.unique()
    all_data.labels_i = all_data.labels_i.replace(label_type[1], 1)
    all_data.labels_i = all_data.labels_i.replace(label_type[0], 0)

    all_labels = all_data.labels_i
    # Remove extra information -----------------------------------------------------
    subject_index = all_data.subject_id__desc
    col_index = find_feat_col(all_data)  # Remove description
    value = all_data[col_index]  # Remove description

    subject_uniq = subject_index.unique()
    test_subjects = subject_uniq[
        np.random.choice(len(subject_uniq), 3, replace=False)]  # Randomly select 3 subjects for testing

    valid_subject = test_subjects[-1]
    test_subjects = test_subjects[0:-1]

    test_subject_index = np.isin(subject_index, test_subjects)
    valid_subject_index = np.array(subject_index == valid_subject)

    test_data = value[test_subject_index]
    test_label = all_labels[test_subject_index]

    valid_data = value[valid_subject_index]
    valid_label = all_labels[valid_subject_index]

    # The remaining subjects will be used for validation
    train_subject_index = ~(test_subject_index | valid_subject_index)
    train_data = value[train_subject_index]
    train_label = all_labels[train_subject_index]

    test_data = reshape_DD(test_data)
    valid_data = reshape_DD(valid_data)
    train_data = reshape_DD(train_data)

    logger.info("{} samples will be used for training".format(len(train_label)))
    logger.info("{} samples will be used for validation".format(len(valid_label)))
    logger.info("{} samples will be used for testing".format(len(test_label)))

    Data['max_len'] = train_data.shape[-1]
    Data['All_train_data'] = np.concatenate([train_data, valid_data])
    Data['All_train_label'] = np.concatenate([train_label.to_numpy(), valid_label.to_numpy()])
    Data['train_data'] = train_data
    Data['train_label'] = train_label.to_numpy()
    Data['val_data'] = valid_data
    Data['val_label'] = valid_label.to_numpy()
    Data['test_data'] = test_data
    Data['test_label'] = test_label.to_numpy()

    return Data


def find_feat_col(data):
    column_list = data.columns.values.tolist()
    filter_col = filter(lambda a: 'feat' in a, column_list)
    index = list(filter_col)
    return index


def clean_Preprocessed_DD(all_data):
    # Drop other class data ------------------------------------------------------------------------------
    noise_class = ['x', 'EyesCLOSEDneutral', 'EyesOPENneutral', 'LateBoredomLap']
    all_labels = all_data.label__desc  # spliting the labels
    all_data = all_data.drop(np.squeeze(np.where(np.isin(all_labels, noise_class))))  # dropping the other classes

    # Added by Saad Irtza ---------------------------------------------------------
    cq_boolean = np.array(all_data['minimum_cq__desc'] > 3)
    all_data = all_data[cq_boolean]
    # ----------------------------------------------------------------------------
    Distraction = all_data.label__desc.unique()
    Distraction = np.setdiff1d(Distraction, 'Driving')
    Distraction = np.setdiff1d(Distraction, 'BoredomLap')
    all_data.label__desc = all_data.label__desc.replace(Distraction, 1)
    all_data.label__desc = all_data.label__desc.replace('Driving', 0)
    all_data.label__desc = all_data.label__desc.replace('BoredomLap', 0)
    return all_data


def reshape_DD(data):
    data = data.values.reshape(data.shape[0], 14, 256)  # Reshape to 2D
    return data


def preprocess_eeg_data(eeg_data, high_freq, low_freq, fs):
    """
    Preprocess EEG data by applying high-pass and low-pass filters and normalization.

    Args:
        eeg_data (np.ndarray): EEG data tensor of shape (num_samples, num_channels, length).
        high_freq (float): High-pass filter cutoff frequency in Hz.
        low_freq (float): Low-pass filter cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.

    Returns:
        preprocessed_data (np.ndarray): Preprocessed EEG data.
    """
    num_samples, num_channels, length = eeg_data.shape
    preprocessed_data = np.zeros_like(eeg_data)

    for channel in range(num_channels):
        # High-pass filter
        b_high, a_high = signal.butter(4, high_freq, btype='high', fs=fs)
        for sample in range(num_samples):
            eeg_data[sample, channel, :] = signal.lfilter(b_high, a_high, eeg_data[sample, channel, :])

        # Low-pass filter
        b_low, a_low = signal.butter(4, low_freq, btype='low', fs=fs)
        for sample in range(num_samples):
            eeg_data[sample, channel, :] = signal.lfilter(b_low, a_low, eeg_data[sample, channel, :])

        # Normalize using min-max scaling
        min_val = np.min(eeg_data[:, channel, :])
        max_val = np.max(eeg_data[:, channel, :])
        preprocessed_data[:, channel, :] = (eeg_data[:, channel, :] - min_val) / (max_val - min_val)

    return preprocessed_data




def select_top_channels_by_importance(train_data, train_label, top_k=16):
    """
    train_data: shape 
    train_label: shape 
    top_k: 
    """
    features = []
    for sample in train_data:
        mean = sample.mean(axis=1)      # 
        std = sample.std(axis=1)        # 
        feat = np.concatenate([mean, std])  # 
        features.append(feat)
    features = np.array(features)  #

    import lightgbm as lgb
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(features, train_label)

    importances = clf.feature_importances_  #
    channel_importance = importances[:features.shape[1]//2] + importances[features.shape[1]//2:]
    top_indices = np.argsort(channel_importance)[::-1][:top_k]

    return top_indices

def select_top_channels_by_importance_from_features(features, labels, top_k=16):
    """
    features: shape 
    labels: shape 
    top_k: 
    """
    clf = lgb.LGBMClassifier(n_estimators=100, random_state=42)
    clf.fit(features, labels)

    importances = clf.feature_importances_  # 
    channel_importance = importances[:features.shape[1]//2] + importances[features.shape[1]//2:]
    top_indices = np.argsort(channel_importance)[::-1][:top_k]

    return top_indices


def load_physio_p300_data(subject_ids=None):

    dataset_fold= "/public/home/liulonglong2024/lon/dataset/PhysioNetP300"
    """

    Args:


    Returns:
       
    """
    if subject_ids is None:
        subject_ids = [1,2, 3, 4, 5, 6, 7, 9, 11]  # 

    data_dict = {}

    # 
    for label in [0, 1]:
        label_path = os.path.join(dataset_fold, str(label))
        if not os.path.exists(label_path):
            print(f"warning:  {label_path} error")
            continue

        files = os.listdir(label_path)
        data_list = []
        subject_list = []

        for file in files:
            # 
            parts = file.split('.')
            if len(parts) > 1 and parts[1].startswith('sub'):
                sub_id = int(parts[1][3:])  # 

                # 
                if subject_ids and sub_id not in subject_ids:
                    continue

                # 
                try:
                    data = torch.load(os.path.join(label_path, file))
                    data_list.append(data)
                    subject_list.append(sub_id)
                except Exception as e:
                    print(f"down load file {file} error: {e}")

        if data_list:
            # 
            data_tensor = torch.stack(data_list)
            data_dict[f"data_{label}"] = data_tensor
            data_dict[f"subjects_{label}"] = subject_list
            print(f" {label} data shape: {data_tensor.shape}")

    return data_dict


def load_data_from_files(root, files, sampling_rate=200):
    
    all_data = []
    all_labels = []
    default_rate = 200
    

    
    for i, file in enumerate(files):
        if i % 100 == 0:
      
            pass
            
        try:
            sample = pickle.load(open(os.path.join(root, file), "rb"))
            X = sample["signal"]
            
            # 
            if sampling_rate != default_rate:
                X = resample(X, 5 * sampling_rate, axis=-1)
            
            # 转换为numpy数组
            X = np.array(X, dtype=np.float32)
            Y = int(sample["label"][0] - 1)
            
            all_data.append(X)
            all_labels.append(Y)
            
        except Exception as e:
            print(f" {file} load error: {e}")
            continue
    
 
    return np.array(all_data, dtype=np.float32), np.array(all_labels)


from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_swat(batch_size,label=False):
    os.chdir("../../data/")
    data = pd.read_csv("swat/SWaT_Dataset_Attack_v0.csv")
    data = data.rename(columns={"Normal/Attack":"label"})
    data = data.rename(columns={" Timestamp":"Timestamp"})
    data['Timestamp'] = data['Timestamp'].str.strip()

    data.label[data.label!="Normal"]=1
    data.label[data.label=="Normal"]=0
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], format='%d/%m/%Y %H:%M:%S %p')
    data = data.set_index("Timestamp")

    feature = data.iloc[:,:51]
    mean_df = feature.mean(axis=0)
    std_df = feature.std(axis=0)

    norm_feature = (feature-mean_df)/std_df
    norm_feature = norm_feature.dropna(axis=1)
    n_sensor = len(norm_feature.columns)

    train_df = norm_feature.iloc[:int(0.6*len(data))]
    train_label = data.label.iloc[:int(0.6*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = data.label.iloc[int(0.6*len(data)):int(0.8*len(data))]
    
    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = data.label.iloc[int(0.8*len(data)):]
    if label:
        train_loader = DataLoader(SWATLabel(train_df,train_label), batch_size=batch_size, shuffle=True)
    else:
        train_loader = DataLoader(SWAT(train_df,train_label), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SWAT(val_df,val_label), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWAT(test_df,test_label), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

class SWAT(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(SWAT, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1), self.label[index], index


class SWATLabel(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10):
        super(SWATLabel, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.label = 1.0-2*self.label 
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)

        delat_time =  df.index[end_idx]-df.index[start_idx]
        idx_mask = delat_time==pd.Timedelta(self.window_size,unit='s')

        return df.values, start_idx[idx_mask], label[start_idx[idx_mask]]

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])

        return torch.FloatTensor(data).transpose(0,1),self.label[index], index

def load_psm(batch_size, window_size, stride_size,train_split,label=False):
    os.chdir("../../data/")
    data = pd.read_csv("psm/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("psm/test_label.csv")
    labels = labels.iloc[:,1]
    data = data.astype(float)
    
    
    feature = data.iloc[:,:25]
    scaler = StandardScaler()
    

    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]

    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:int(0.60*len(data))]
    train_label = labels[:int(0.60*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8*len(data))]
    

    test_df = norm_feature.iloc[int(0.80*len(data)):]
    test_label = labels[int(0.80*len(data)):]

    if label:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PSM(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PSM(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


class PSM(Dataset):
    def __init__(self, df, label, window_size=60, stride_size=10) -> None:
        super(PSM, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size,self.stride_size)
        end_idx = np.arange(self.window_size, len(df), self.stride_size)
        
    
        label = [0 if sum(label[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        return df.values, start_idx, np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length

    def __getitem__(self, index):
        #  N X K X L X D 
        """
        """
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index], index

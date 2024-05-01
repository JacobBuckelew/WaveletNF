import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np


def loader_PSM(batch_size, window_size, stride_size,train_split,label=False):
    data = pd.read_csv("psm/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("psm/test_label.csv")
    labels = labels.iloc[:,1].values
    data = data.astype(float)
    
    #%%
    
    feature = data.iloc[:,:25]
    scaler = StandardScaler()
    

    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]
    print("num sensors:", n_sensor)

    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:int(0.60*len(data))]
    train_label = labels[:int(0.60*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8*len(data))]
    

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.8*len(data)):]

    if label:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PSM(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PSM(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor

def loader_PSM_OCC(root, batch_size, window_size, stride_size,train_split,label=False):
    
    data = pd.read_csv("PSM/train.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    
    labels = [0] * len(data)
    data = data.astype(float)
    
    #%%
    print(data.shape)
    feature = data.iloc[:,:25]
    print(feature.shape)


    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]


    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)

    norm_feature = norm_feature.dropna(axis=0)

 
    train_df = norm_feature.iloc[:]
    train_label = labels[:]

    val_df = norm_feature.iloc[int(train_split*len(data)):]
    val_label = labels[int(train_split*len(data)):]
    
    
    
    
    data = pd.read_csv("PSM/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("PSM/test_label.csv")
    labels = labels.iloc[:,1].values
    data = data.astype(float)
    
    #%%
    
    feature = data.iloc[:,:25]
    
   
    scaler = StandardScaler()
    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]

 
    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    test_df = norm_feature.iloc[int(train_split*len(data)):]
    test_label = labels[int(train_split*len(data)):]

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

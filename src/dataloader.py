import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler

def load_swat(window_size, stride, batch_size, val_split=0.80):
    
    data = pd.read_csv("../data/swat/SWaT_Dataset_Attack_v0.csv")

    data = data.rename(columns={" Timestamp":"Timestamp"})
    data['Timestamp'] = data['Timestamp'].str.strip()
    Timestamp_tr = pd.to_datetime(data["Timestamp"])
    data["Timestamp"] = Timestamp_tr
    data = data.set_index("Timestamp")
    data = data.rename(columns={"Normal/Attack":"label"})
    data.label[data.label!="Normal"]=1
    data.label[data.label=="Normal"]=0

    data = data.astype(float)
    feature = data.iloc[:,:51]
    scaler = MinMaxScaler()
    
    norm_feature = scaler.fit_transform(feature)

    norm_feature = pd.DataFrame(norm_feature, columns= feature.columns, index = Timestamp_tr)
    norm_feature = norm_feature.dropna(axis=1)
    test_label = data.label.iloc[int(0.8*len(data)):]
    n_sensor = norm_feature.shape[1]

    train_df = norm_feature.iloc[:int(0.60 * len(norm_feature))]
    val_df = norm_feature.iloc[int(0.60 * len(norm_feature)):int(val_split * len(norm_feature))]
    test_df = norm_feature.iloc[int(val_split * len(norm_feature)):]
    train_loader = DataLoader(SWAT(train_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SWAT(val_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWAT(test_df,test_label, window_size, stride), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

def load_data(dataset, window_size, stride, batch_size):

    if dataset == "SWAT":
        train_loader, val_loader, test_loader, n_sensor = load_swat(window_size, stride, batch_size)

    elif dataset == "PSM":
        train_loader, val_loader, test_loader, n_sensor = load_psm(window_size, stride, batch_size)
    
    else:
        raise Exception(f"{dataset} is not a valid dataset option.")

    return train_loader, val_loader, test_loader, n_sensor



class PSM(Dataset):

    def __init__(self, df, labels, window_size, stride_size):
        super(PSM, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df, labels)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]

    def preprocess(self, df, labels):

        start_idx = np.arange(0, len(df) - self.window_size + 1, self.stride_size)
        if labels is not None:
            label = [0 if sum(labels[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        else:
            label = [0 for index in start_idx]
        return df.values, start_idx, np.array(label)

    def __len__(self):
        return len(self.idx)

    
    def __getitem__(self, index):
        # T X N X D
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index]

def load_psm(window_size, stride_size, batch_size,label=False):
    data = pd.read_csv("../data/psm/test.csv")
    Timestamp = pd.to_datetime(data["timestamp_(min)"])
    data["Timestamp"] = Timestamp
    data = data.set_index("Timestamp")
    labels = pd.read_csv("../data/psm/test_label.csv")
    labels = labels.iloc[:,1].values
    data = data.astype(float)
    
    
    feature = data.iloc[:,:25]
    scaler = MinMaxScaler()
    

    norm_feature = scaler.fit_transform(feature)

    n_sensor = norm_feature.shape[1]

    norm_feature = pd.DataFrame(norm_feature, columns= data.columns[1:], index = Timestamp)
    norm_feature = norm_feature.dropna(axis=1)
    train_df = norm_feature.iloc[:int(0.60*len(data))]
    train_label = labels[:int(0.60*len(data))]

    val_df = norm_feature.iloc[int(0.60*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8 * len(data))]

    test_df = norm_feature.iloc[int(0.8*len(data)):]
    test_label = labels[int(0.80*len(data)):]

    if label:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    else:
        train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PSM(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PSM(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


class PSM(Dataset):
    def __init__(self, df, label, window_size, stride_size=10) -> None:
        super(PSM, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        #self.label = 1.0 - self.label
    
    def preprocess(self, df, label):

        start_idx = np.arange(0,len(df)-self.window_size + 1,self.stride_size)
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
        return torch.FloatTensor(data).transpose(0,1), self.label[index]


class SWAT(Dataset):

    def __init__(self, df, labels, window_size, stride_size):
        super(SWAT, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size

        self.data, self.idx, self.label = self.preprocess(df, labels)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]

    def preprocess(self, data, labels):

        start_idx = np.arange(0, len(data) - self.window_size + 1, self.stride_size)
        if labels is not None:
            label = [0 if sum(labels[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        else:
            label = [0 for index in start_idx]
        return data.values, start_idx, np.array(label)

    def __len__(self):
        return len(self.idx)

    
    def __getitem__(self, index):
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index]
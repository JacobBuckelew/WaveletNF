import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import os


def load_rtds(window_size, stride_size, batch_size):
    train_split = 0.80

    train_df = pd.read_csv("../data/pmu/training/data.csv")
    train_df = train_df.set_index("Timestamp")
    train_df = train_df.drop(train_df.columns[[0]], axis=1)
    print("data length:", len(train_df))
    scaler = MinMaxScaler()
    idx = train_df.index
    norm_train = pd.DataFrame(scaler.fit_transform(train_df))
    norm_train.index = idx
    print(norm_train.shape)
    train_df = norm_train.iloc[:int(train_split * len(train_df))]
    n_sensor = norm_train.shape[1]
    val_df = norm_train.iloc[int(train_split * len(train_df)):]

    train_loader = DataLoader(PMU(df=train_df, labels=None, window_size=window_size, stride_size=10), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(PMU(df=val_df, labels=None, window_size=window_size, stride_size=10), batch_size=batch_size, shuffle=False)
    test_loaders = []
    runs = 5
    total_labels = []
    total = 0
    for i in range(runs):
        
        df = pd.read_csv(f"../data/pmu/test/run{i+1}/data.csv")
        df = df.set_index("Timestamp")
        T = len(df)
        total += T
        idx = df.index
        norm_df = pd.DataFrame(scaler.transform(df))
        norm_df.index = idx
        # load labels
        labels = pd.read_csv(f"../data/pmu/test/run{i+1}/labels.csv")
        labels = labels.drop(labels.columns[[0]], axis=1)
        label = [0 if sum(labels.iloc[index]) == 0 else 1 for index in range(len(labels))]
        total_labels.append(label)
        loader = DataLoader(PMU(df=norm_df, labels=labels, window_size=window_size, stride_size=10), batch_size=batch_size, shuffle=False)
        test_loaders.append(loader)
    #print("len test set:", T)
    total_label = np.concatenate(total_labels)
    #print("anomaly ratio in test set:", np.sum(total_label)/len(total_label))
    return train_loader, val_loader, test_loaders, n_sensor
    


def load_wadi(window_size, stride_size, batch_size):
    data = pd.read_csv("../data/wadi/WADI_attackdata.csv")
    labels=[]
    train_split = 0.60

    for index, row in data.iterrows():
        date_temp=row['Date']
        date_mask="%m/%d/%Y"
        date_obj=datetime.strptime(date_temp, date_mask)
        time_temp=row['Time']
        time_mask="%I:%M:%S.%f %p"
        time_obj=datetime.strptime(time_temp,time_mask)

        if date_obj==datetime.strptime('10/9/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('7:25:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('7:50:16.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        if date_obj==datetime.strptime('10/10/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('10:24:10.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:34:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:24:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:30:40.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:44:50.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('1:39:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('1:50:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('2:48:17.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('2:59:55.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('5:40:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('5:49:40.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('10:55:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('10:56:27.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
        
        if date_obj==datetime.strptime('10/11/2017', '%m/%d/%Y'):
            if time_obj>=datetime.strptime('11:17:54.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:31:20.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:36:31.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('11:47:00.000 AM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('11:59:00.000 AM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:05:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:07:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:10:52.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('12:16:00.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('12:25:36.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue
            elif time_obj>=datetime.strptime('3:26:30.000 PM', '%I:%M:%S.%f %p') and time_obj<=datetime.strptime('3:37:00.000 PM', '%I:%M:%S.%f %p'):
                labels.append('Attack')
                continue

        labels.append('Normal')
 
    Timestamp = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data=data.drop(data.columns[[0,1,2,50,51,86,87]],axis=1) 
    labels = [ int(l!= 'Normal' ) for l in labels]

    data = data.astype(float)

    n_sensor = len(data.columns)

    feature = data
    scaler = MinMaxScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_feature = pd.DataFrame(norm_feature, index = Timestamp, columns=data.columns)
    norm_feature = norm_feature.dropna(axis=0)

    train_df = norm_feature.iloc[:int(train_split*len(data))]
    train_label = labels[:int(train_split*len(data))]

    val_df = norm_feature.iloc[int(0.6*len(data)):int(0.8*len(data))]
    val_label = labels[int(0.6*len(data)):int(0.8*len(data))]

    test_df = norm_feature.iloc[int(0.80*len(data)):]
    test_label = labels[int(0.80*len(data)):]

    print("Train data:", len(train_df))
    print("Test data:", len(test_df))

    train_loader = DataLoader(WADI(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(WADI(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor



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
    test_label = labels[int(0.8*len(data)):]

    


    print("Train data:", len(train_df))
    print("Test data:", len(test_df))


    
    train_loader = DataLoader(PSM(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PSM(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PSM(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


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
    print("n_sensor:", n_sensor)

    train_df = norm_feature.iloc[:int(0.60 * len(norm_feature))]
    val_df = norm_feature.iloc[int(0.60 * len(norm_feature)):int(val_split * len(norm_feature))]
    test_df = norm_feature.iloc[int(val_split * len(norm_feature)):]
    train_loader = DataLoader(SWAT(train_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SWAT(val_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWAT(test_df,test_label, window_size, stride), batch_size=batch_size, shuffle=False)
    print("Train data:", len(train_df))
    print("Test data:", len(test_df))


    return train_loader, val_loader, test_loader, n_sensor



import pickle
def load_smd(dataset, window_size = 60, stride_size = 10, batch_size= 256, train_split = 0.6, do_preprocess=True, train_start=0,
             test_start=0):
    """
    get data from pkl files

    return shape: (([train_size, x_dim], [train_size] or None), ([test_size, x_dim], [test_size]))
    """
    prefix = "../data/smd/"
    x_dim = 38
 
    try:
        f = open(os.path.join(prefix, dataset + '_test.pkl'), "rb")
        test_data = pickle.load(f).reshape((-1, x_dim))[test_start:, :]
        f.close()
    except (KeyError, FileNotFoundError):
        print("Data not found")
        test_data = None
    prefix = "../data/smd/labels"
    try:
        f = open(os.path.join(prefix, dataset + "_test_label.pkl"), "rb")
        test_label = pickle.load(f).reshape((-1))[test_start:]
        f.close()
    except (KeyError, FileNotFoundError):
        print("Labels not found")
        test_label = None

    whole_data = test_data
    whole_label = test_label
    if do_preprocess:
        whole_data = preprocess(whole_data)
   
    n_sensor = whole_data.shape[1]
    print(len(whole_label))



    train_df = whole_data[:int(train_split*len(whole_data))]
    train_label = whole_label[:int(train_split*len(whole_data))]


    val_df = whole_data[int(0.6*len(whole_data)):int(0.8*len(whole_data))]
    val_label = whole_label[int(0.6*len(whole_data)):int(0.8*len(whole_data))]

    test_df = whole_data[int(0.8*len(whole_data)):]
    test_label = whole_label[int(0.8*len(whole_data)):]

    print("Train data:", len(train_df))
    print("Test data:", len(test_df))


    train_loader = DataLoader(SMD(train_df,train_label, window_size, stride_size), batch_size=batch_size, shuffle=False)

    val_loader = DataLoader(SMD(val_df,val_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SMD(test_df,test_label, window_size, stride_size), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor


def preprocess(df, mode = 'Normal'):
    """returns normalized and standardized data.
    """

    df = np.asarray(df, dtype=np.float32)


    if len(df.shape) == 1:
        raise ValueError('Data must be a 2-D array')

    if np.any(sum(np.isnan(df)) != 0):
        df = np.nan_to_num()

  
    df = MinMaxScaler().fit_transform(df)

    return df


def load_data(dataset, window_size, stride, batch_size):

    if dataset == "SWAT":
        train_loader, val_loader, test_loader, n_sensor = load_swat(window_size, stride, batch_size)
    elif dataset == "PSM":
        train_loader, val_loader, test_loader, n_sensor = load_psm(window_size, stride, batch_size)
    elif dataset.startswith('machine'):
        train_loader, val_loader, test_loader, n_sensor = load_smd(dataset, window_size, stride, batch_size)
    elif dataset == "WADI":
        train_loader, val_loader, test_loader, n_sensor = load_wadi(window_size, stride, batch_size)
    elif dataset == "PMU":
        train_loader, val_loader, test_loader, n_sensor = load_rtds(window_size, stride, batch_size)
    else:
        raise Exception(f"{dataset} is not a valid dataset option.")


    return train_loader, val_loader, test_loader, n_sensor


class PSM(Dataset):
    def __init__(self, df, label, window_size, stride_size) -> None:
        super(PSM, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.labels = label
        print(len(self.labels))
    
        self.data, self.idx, self.label = self.preprocess(df,label)
        self.columns = np.append(df.columns, ["Label"])
        self.timeindex = df.index[self.idx]
        #self.label = 1.0 - self.label
    
    def preprocess(self, df, label):


        start_idx = np.arange(0,len(df) - self.window_size,self.stride_size)

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
        self.labels = labels
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

class SMD(Dataset):
    def __init__(self, df, label, window_size, stride_size=10) -> None:
        super(SMD, self).__init__()
        self.df = df
        self.labels = label
        self.window_size = window_size
        self.stride_size = stride_size
        self.data, self.idx, self.label = self.preprocess(df,label)
        #self.timeindex = df.index[self.idx]

        #self.label = 1.0 - self.label
    def preprocess(self, df, labels):

        start_idx = np.arange(0,len(df)-self.window_size + 1,self.stride_size)
        if labels is not None:
            label = [0 if sum(labels[index:index+self.window_size]) == 0 else 1 for index in start_idx]
        else:
            label = [0 for index in start_idx]
        return df, start_idx, np.array(label)

    def __len__(self):

        length = len(self.idx)

        return length   

    def __getitem__(self, index):
        #  N X K X L X D 

        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size,-1, 1])
        return torch.FloatTensor(data).transpose(0,1), self.label[index]


class WADI(Dataset):

    def __init__(self, df, labels, window_size, stride_size):
        super(WADI, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.labels = labels
        self.data, self.idx, self.label = self.preprocess(df, labels)
        self.columns = np.append(df.columns, ["Label"])

        self.timeindex = df.index[self.idx]
        print(self.timeindex)

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

class PMU(Dataset):

    def __init__(self, df, labels, window_size, stride_size):
        super(PMU, self).__init__()
        self.df = df
        self.window_size = window_size
        self.stride_size = stride_size
        self.labels = labels
        self.data, self.idx, self.label = self.preprocess(df, labels)
        self.columns = np.append(df.columns, ["Label"])

        self.timeindex = df.index[self.idx]

    def preprocess(self, data, labels):

        start_idx = np.arange(0, len(data) - self.window_size + 1, self.stride_size)

        if labels is not None:
            labels = labels.drop(labels.columns[[0]], axis=1)
            label = [1 if  (labels[index:index+self.window_size].sum() > 0).any() else 0 for index in start_idx]
            #print(label)
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


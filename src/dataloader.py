import torch
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler



def load_swat(window_size, stride, batch_size, val_split=0.80):
    
    test_data = pd.read_csv("../data/swat/SWaT_Dataset_Attack_v0.csv")
    train_data = pd.read_csv("../data/swat/SWaT_Dataset_Normal_v0.csv", skiprows=1)

    train_data = train_data.rename(columns={" Timestamp":"Timestamp"})
    test_data = test_data.rename(columns={" Timestamp": "Timestamp"})
    train_data['Timestamp'] = train_data['Timestamp'].str.strip()
    test_data['Timestamp'] = test_data["Timestamp"].str.strip()
    Timestamp_tr = pd.to_datetime(train_data["Timestamp"])
    Timestamp_test = pd.to_datetime(test_data["Timestamp"])
    train_data["Timestamp"] = Timestamp_tr
    train_data = train_data.set_index("Timestamp")
    test_data["Timestamp"] = Timestamp_test
    test_data = test_data.set_index("Timestamp")

    labels = [ int(l!= 'Normal' ) for l in test_data["Normal/Attack"].values]
    for i in list(test_data): 
        test_data[i]=test_data[i].apply(lambda x: str(x).replace("," , "."))
    test_data = test_data.drop(["Normal/Attack"] , axis = 1)
    train_data = train_data.drop(["Normal/Attack"], axis = 1)
    train_data, test_data = train_data.astype(float), test_data.astype(float)
    
    feature = train_data.iloc[:,:51]
    test_features = test_data.iloc[:, :51]
    print(feature.shape)
    print(test_features.shape)
    scaler = MinMaxScaler()
    
    norm_feature = scaler.fit_transform(feature)
    norm_test = scaler.fit(test_features)

    norm_feature = pd.DataFrame(norm_feature, columns= train_data.columns, index = Timestamp_tr)
    norm_feature = norm_feature.dropna(how="all")
    norm_feature.fillna(0, inplace=True)
    n_sensor = norm_feature.shape[1]
    norm_test = pd.DataFrame(norm_test, columns= test_data.columns, index = Timestamp_test)
    norm_test = norm_test.dropna(how="all")
    norm_test.fillna(0, inplace=True)

    train_df = norm_feature.iloc[:int(val_split*len(norm_feature))]
    val_df = norm_feature.iloc[int(val_split * len(norm_feature)):]
  
    train_loader = DataLoader(SWAT(train_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(SWAT(val_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(SWAT(norm_test,labels, window_size, stride), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor

def load_wadi(window_size, stride, batch_size, val_split=0.80):
    # train data

    train_data = pd.read_csv("../data/wadi/train.csv", skiprows=1000)
    col = ["Row", "Date", "Time"] + [f"Sensor{i}" for i in range(len(train_data.columns)- 3)]
    train_data.columns = col
    Timestamp_tr = pd.to_datetime(train_data['Date'] + ' ' + train_data['Time'])
    #train_data.dropna(how='all', inplace=True)
    #train_data.fillna(0, inplace=True)
    train_data=train_data.drop(train_data.columns[[0,1,2,50,51,86,87]],axis=1) 

    # test data

    data = pd.read_csv("../data/wadi/test.csv",sep=",")
    labels=[]
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

    train_data, data = train_data.astype(float), data.astype(float)

    n_sensor = train_data.shape[1]

    feature = train_data
    scaler = MinMaxScaler()
    norm_feature = scaler.fit_transform(feature)
    norm_test = scaler.fit(data)

    norm_feature = pd.DataFrame(norm_feature, columns= train_data.columns, index = Timestamp_tr)
    n_sensor = norm_feature.shape[1]
    norm_feature.dropna(how='all', inplace=True)
    norm_feature.fillna(0, inplace=True)
    norm_test = pd.DataFrame(norm_test, columns= data.columns, index = Timestamp)
    norm_test = norm_test.dropna(how="all")
    norm_test.fillna(0, inplace=True)

    train_df = norm_feature.iloc[:int(val_split*len(norm_feature))]
    val_df = norm_feature.iloc[int(val_split * len(norm_feature)):]
  
    train_loader = DataLoader(WADI(train_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(WADI(val_df,None, window_size, stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WADI(norm_test,labels, window_size, stride), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, n_sensor



def load_psm(window_size, stride, batch_size, val_split=0.80):

    train_data = pd.read_csv("../data/psm/train.csv")
    Timestamp_tr = pd.to_datetime(train_data["timestamp_(min)"])
    train_data["Timestamp"] = Timestamp_tr
    train_data = train_data.set_index("Timestamp")
    test_data = pd.read_csv("../data/psm/test.csv")
    Timestamp_test = pd.to_datetime(test_data["timestamp_(min)"])
    test_data["Timestamp"] = Timestamp_test
    test_data = test_data.set_index("Timestamp")
    labels = pd.read_csv("../data/psm/test_label.csv")
    labels = labels.iloc[:,1].values
    train_data, test_data = train_data.astype(float), test_data.astype(float)
    
    feature = train_data.iloc[:,:25]
    test_features = test_data.iloc[:,:25]
    scaler = MinMaxScaler()
    
    norm_feature = scaler.fit_transform(feature)
    norm_test = scaler.transform(test_features)

    norm_feature = pd.DataFrame(norm_feature, columns= train_data.columns[1:], index = Timestamp_tr)
    norm_feature = norm_feature.dropna(how="all")
    norm_feature.fillna(0, inplace=True)
    n_sensor = norm_feature.shape[1]
    norm_test = pd.DataFrame(norm_test, columns= test_data.columns[1:], index = Timestamp_test)
    norm_test = norm_test.dropna(how="all")
    norm_test.fillna(0, inplace=True)

    train_df = norm_feature.iloc[:int(val_split*len(train_data))]
    val_df = norm_feature.iloc[int(val_split*len(train_data)):]
    train_loader = DataLoader(PSM(train_df, labels = None, window_size=window_size, stride_size=stride), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(PSM(val_df,labels=None, window_size=window_size, stride_size=stride), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(PSM(norm_test,labels, window_size, stride), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, n_sensor

def load_data(dataset, window_size, stride, batch_size):

    if dataset == "SWaT":
        train_loader, val_loader, test_loader, n_sensor = load_swat(window_size, stride, batch_size)

    elif dataset == "WADI":
        train_loader, val_loader, test_loader, n_sensor = load_wadi(window_size, stride, batch_size)

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


class WADI(Dataset):

    def __init__(self, df, labels, window_size, stride_size):
        super(WADI, self).__init__()
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
        # T X N X D
        start = self.idx[index]
        end = start + self.window_size
        data = self.data[start:end].reshape([self.window_size, -1, 1])
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
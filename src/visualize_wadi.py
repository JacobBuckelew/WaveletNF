# %%
# load data 
from datetime import datetime
import pandas as pd

# load data
data = pd.read_csv("data/wadi/WADI_attackdata.csv")
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

train_df = data.iloc[:int(train_split*len(data))]
train_label = labels[:int(train_split*len(data))]

val_df = data.iloc[int(0.6*len(data)):int(0.8*len(data))]
val_label = labels[int(0.6*len(data)):int(0.8*len(data))]

test_df = data.iloc[int(0.80*len(data)):]
test_label = labels[int(0.80*len(data)):]

# %%
# load likelihoods
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)
        
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_json(file):
    with open(file) as f:
        obj = json.load(f)
    return obj



def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NpEncoder, indent=4)

test_results = load_json("figures/wadi_likelihoods")

likelihoods = test_results["likelihoods"]
timestamp_labels = test_results["labels"]

# %%

#print(test_df.columns)

# %%
print(Timestamp[147295:147390])

# %%
print(test_df.loc[163590:163700])
subset_df = test_df.iloc[:, [18, 19, 38]]
# 38

# %%

subset_df.columns

# %%
subset_df.loc[147280:147400].plot()

# %%
likelihood_df = pd.DataFrame({'values': likelihoods}, index=test_df.index[1:])
likelihood_df = likelihood_df * -1
likelihood_df.head

# %%
likelihood_df['values'] = (likelihood_df['values'] - likelihood_df['values'].min()) / (likelihood_df['values'].max() - likelihood_df['values'].min())

# %%
likelihood_df.loc[147280:147400].plot()

# %%
subset_df = subset_df.loc[147280:147400]

# %%
likelihood_df = likelihood_df.loc[147280:147400]

# %%
subset_df["likelihoods"] = likelihood_df.iloc[:, 0]


# %%
subset_df.columns = list(subset_df.columns)
subset_df.columns = ['P_006', 'DPIT_001_PV', 'FIC_601_CO', 'likelihoods']

# %%
subset_df.columns

# %%
subset_df["timestamp"] = Timestamp.loc[147280:147400]

# %%
subset_df.columns

# %%
#subset_df.set_index('timestamp', inplace=True)

# %%
subset_df.head

# %%
steps = [i for i in range(len(subset_df))]
subset_df["steps"] = steps

# %%
subset_df.set_index("steps", inplace=True)

# %%
subset_df.head

# %%
from matplotlib import pyplot as plt

# Plotting
fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
subset_df.columns = ["Pump Status", "Pressure", "Flow", "Likelihoods", "time"]
for i, column in enumerate(subset_df.columns[:-1]):  # Exclude 'timestamp' and 'steps'
    subset_df[column].plot(ax=axes[i])
    axes[i].set_ylabel(column)
subset_df.to_csv("figures/wadi_data.csv")
axes[-1].set_xlabel('Index')  # Label for the bottom plot
plt.savefig("figures/wadi_example.png")
plt.tight_layout()
plt.show()


# %%




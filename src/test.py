import os
os.chdir("../")
import torch
import torch.optim as optim
import argparse
from dataloader import *
from model.model import *
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import *
def get_args():

    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("--checkpt", type=str, default='../checkpoint', help="checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log", type=int, default=20, help="How often to log model")
    parser.add_argument("--log_output", type=str, default="../log")
    parser.add_argument("--name", type=str, default="WaveletNF")
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--dataset", type=str, default="PSM")
    # model parameters for CANF
    parser.add_argument("--st_units", type=int, default=32)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--st_layers", type=int, default=1)
    parser.add_argument("--num_blocks", type=int, default=8)
    parser.add_argument("--wavelet_type", type=str, default="haar")
    parser.add_argument("--wavelet", type=int, default=1)
    parser.add_argument("--attention", type=int, default=1)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--example", type=int, default=0)
    parser.add_argument("--momentum", type=float, default=0.95)
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride_size", type=int, default=10)
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()
    return args

def l1_loss(model, lam):
    loss = 0
    for i in range(len(model.flow)):
        loss += torch.norm(model.flow[i].encoder.attention.W_o.weight, p=1)

    #print("loss:", loss)
    return lam * loss

def save_attention(attention):
    attention = attention[0][0]
    #print(attention)
    path = "../figures"
    A = attention.cpu().numpy()
    A = A[:14, :14]
    #print(A)
    plt.rcParams['ytick.labelsize'] = 20
    X_label = [f'timestamp_{x}' for x in range(0,13)]
    Y_label = [f"sensor_{x}" for x in range(0,13)]
    fig, ax = plt.subplots(figsize=(36, 40)) # set figure size
    heatmap = ax.pcolor(A, cmap="autumn", alpha=0.9)
    fig.colorbar(heatmap)
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]
    
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45, fontsize=20)   # labels should be 'unicode'
    
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label[::-1], minor = False, fontsize=20)   # labels should be 'unicode'
    # Set axis labels
    #if X_label != None and Y_label != None:
        #X_label = [x_label for x_label in X_label]
    #Y_label = [y_label for y_label in Y_label]
    
    #xticks = range(0,len(X_label))
    #ax.set_xticks(xticks, minor=False) # major ticks
    #ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    
    #yticks = range(0,len(Y_label))
    #ax.set_yticks(yticks, minor=False)
    #ax.set_yticklabels(Y_label[::-1], minor = False)   # labels should be 'unicode'
    
    ax.grid(True)
    #plt.show()
    plt.savefig(path + "/attention_matrix.jpg")



def save_attention_scores(attention_scores):
    attention = attention_scores[0][0][0].squeeze()
    #print(attention)
    path = "../figures"
    A = attention.cpu().numpy()
    A = A[1:13, 1:13]
    #print(A)
    plt.rcParams['ytick.labelsize'] = 20
    X_label = [f'sensor_{x}' for x in range(0,12)]
    Y_label = [f"sensor_{x}" for x in range(0,12)]
    fig, ax = plt.subplots(figsize=(24, 26)) # set figure size
    heatmap = ax.pcolor(A, cmap="autumn", alpha=0.9)
    fig.colorbar(heatmap)
    X_label = [x_label for x_label in X_label]
    Y_label = [y_label for y_label in Y_label]
    
    xticks = range(0,len(X_label))
    ax.set_xticks(xticks, minor=False) # major ticks
    ax.set_xticklabels(X_label, minor = False, rotation=45, fontsize=20)   # labels should be 'unicode'
    
    yticks = range(0,len(Y_label))
    ax.set_yticks(yticks, minor=False)
    ax.set_yticklabels(Y_label[::-1], minor = False, fontsize=20)   # labels should be 'unicode'
    # Set axis labels
    #if X_label != None and Y_label != None:
        #X_label = [x_label for x_label in X_label]
    #Y_label = [y_label for y_label in Y_label]
    
    #xticks = range(0,len(X_label))
    #ax.set_xticks(xticks, minor=False) # major ticks
    #ax.set_xticklabels(X_label, minor = False, rotation=45)   # labels should be 'unicode'
    
    #yticks = range(0,len(Y_label))
    #ax.set_yticks(yticks, minor=False)
    #ax.set_yticklabels(Y_label[::-1], minor = False)   # labels should be 'unicode'
    

    ax.grid(True)
    import pandas as pd
    df  = pd.DataFrame(A)
    df.to_csv("../figures/attention_scores.csv")
    #plt.show()
    plt.savefig(path + "/attention_scores.jpg")



if __name__ == "__main__":
    # get args
    args = get_args()
    print(vars(args))
    if args.gpu:
        torch.cuda.manual_seed_all(args.seed)
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
    else:
        device = torch.device("cpu")
    print("testing on ",device)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # load data

    train_loader, val_loader, test_loader, n_sensors = load_data(dataset=args.dataset, window_size=args.window_size, stride=args.stride_size, batch_size=args.batch_size)

    # train model

    # create instance of CANF model
    wavenf = WaveletEnhancedNF(num_blocks=args.num_blocks,
                               hidden_d = args.st_units, 
                               wavelet_type = args.wavelet_type,
                               N = args.N,
                               n_heads = args.heads,
                               num_features = 1, 
                               st_units = args.st_units, 
                               st_layers = args.st_layers,
                               num_entities=n_sensors,
                               wavelet=args.wavelet,
                               attention=args.attention,
                               b_norm= True, 
                               momentum=0.95)
    wavenf = wavenf.to(device)
    checkpt_path = os.path.join(args.checkpt, args.name)
    checkpt_path = checkpt_path + "/best_params.pt"
    print("Loading weights from ", checkpt_path)
    
    wavenf.load_state_dict(torch.load(checkpt_path))
    log_path = os.path.join(args.log_output, args.name)
    print("Logging results to ", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    attention = []
    loss_test = []
    test_labels = []
    scores = []
    inference_times = []
    i = 0



    #print(test_loader.dataset.labels)
    with torch.no_grad():
        wavenf.eval()
        if args.dataset != "PMU":
            for x, labels  in test_loader:
                x = x.to(device)
                if args.example == 1 and args.dataset == "machine-1-2" and args.seed == 6:
                    loss = -1 * wavenf.density_t(x, take_mean=False).cpu().numpy()
                    loss = loss.reshape(-1)
                else:
                    start = timeit.default_timer()
                    loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                    if not(args.attention == 0 or args.lam == 0):
                        loss += l1_loss(wavenf, args.lam).cpu().numpy()
                    end = timeit.default_timer()
                    inference_times.append(end - start)
                    loss_test.append(loss)
                test_labels.append(labels)
                idx = test_loader.dataset.idx[i]

                # SMD running example
                if args.example == 1 and i ==20  and args.dataset == "machine-1-2" and args.seed == 6:
                    attention.append(wavenf.flow[1].attention)
                    scores.append(wavenf.get_attention())
                i = i+1
        else:
            for loader in test_loader:
                for x, labels,  in loader:
                        x = x.to(device)
                        start = timeit.default_timer()
                        loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                        #print(loss.shape)
                        if not(args.attention == 0 or args.lam == 0.0):
                            loss += l1_loss(wavenf, lam=args.lam).cpu().numpy()
                        end = timeit.default_timer()
                        inference_times.append(end - start)
                        #print(l1.shape)
                        loss_test.append(loss)
                        test_labels.append(labels)

    # SMD running example, this is hard-coded for easy reproducibility
    if args.example == 1:
        # save attention
        save_attention(attention)
        save_attention_scores(scores)

        # get time series data for specific window in the example
        # indices for start and end times
        start = 1260
        end = 1305

        df = test_loader.dataset.df[start:end]
        df = pd.DataFrame(df, dtype=float)

        sensor_12 = df.iloc[:, 11].values

        # choose important sensors for visualization
        #print(df.columns)
        #print(df.shape)
        cols = [0, 1,2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        df = df[cols]
        df.columns = [f"Sensor_{i+1}" for i in cols]
        #print(df.head)
        df.to_csv("../figures/running_example_values.csv")



        fig = plt.figure(figsize=(16, 20))

        plt.plot(loss)
        plt.savefig("../figures/running_example_loss.jpg")

        df = pd.DataFrame(loss)
        df.to_csv("../figures/running_example_loss.csv")

        fig = plt.figure(figsize=(12,12))
        plt.plot(sensor_12)
        plt.savefig("../figures/running_example_sensor12.jpg")

        df = pd.DataFrame(sensor_12)
        df.to_csv("../figures/running_example_sensor12.csv")





        #fig = plt.figure(figsize=(20,12))
        #colors = ["blue", "green", "brown", "purple", "red"]
        #plt.legend()

        #j = 0
        #for sensor in df.columns:
            #plt.plot(df[sensor], linestyle="-",label=sensor, color=colors[j], linewidth=5)
            #j +=1

        #plt.legend(fontsize=18)
        #plt.ylabel("Value", fontsize=25)
        #plt.xlabel("Timestamp", fontsize=25)
        #plt.xticks(fontsize=20)
        # beginning of anomaly
        #plt.axvline(x=20, color='r', linestyle='--', linewidth=4, label="Anomaly Start")
        #plt.yticks(fontsize=20)
        #plt.legend(fontsize=18)
        #plt.savefig("../figures/running_example_series.jpg", dpi=400)
        



        # plot assigned likelihood for labeled data

        
    else:
        loss_test = np.concatenate(loss_test)
        test_labels = np.concatenate(test_labels)
        #print(len(loss_test))
        #timestamp_labels = test_loader.dataset.labels[:len(loss_test)]
        #print(len(timestamp_labels))
        #results = pd.DataFrame([loss_test, timestamp_labels]).T
        #esults.to_csv("loss.csv")

        # get final metrics
        print("Getting final metrics")
        results = get_metrics(loss_test, test_labels)
        num_params = sum(p.numel() for p in wavenf.parameters() if p.requires_grad)
        results["Log-Density"]= np.mean(loss_test)
        results["Inference_Time"] = np.mean(inference_times)
        results["Parameters"] = num_params
        save_json(results, f'{log_path}/test_results')
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
from utils import save_json
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
    parser.add_argument("--momentum", type=float, default=0.95)
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride_size", type=int, default=10)
    args = parser.parse_args()
    return args


def save_attention(attention_scores):
    print(attention_scores[0][0].shape)
    attention = attention_scores[0][0][1].squeeze()
    print(os.getcwd())
    path = "../figures"
    A = attention.cpu().numpy()
    plt.rcParams['ytick.labelsize'] = 20
    X_label = [f'feature_{x}' for x in range(1,26)]
    Y_label = [f"feature_{x}" for x in range(1,26)]
    fig, ax = plt.subplots(figsize=(16, 14)) # set figure size
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
    plt.savefig(path + "/attention_matrix_2.jpg")



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

    train_loader, val_loader, test_loader, n_sensors = load_data(dataset=args.dataset, window_size=args.window_size, stride=10, batch_size=args.batch_size)

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
    inference_times = []
    i = 0
    with torch.no_grad():
        wavenf.eval()
        for x, labels  in test_loader:
            x = x.to(device)
            start = timeit.default_timer()
            loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
            end = timeit.default_timer()
            if i == 105:
                attention.append(wavenf.get_attention())
                break
            inference_times.append(end - start)
            loss_test.append(loss)
            test_labels.append(labels)
            i = i+1
            
    loss_test = np.concatenate(loss_test)
    test_labels = np.concatenate(test_labels)
    test_auc = roc_auc_score(test_labels, loss_test)
    save_attention(attention)
    num_params = sum(p.numel() for p in wavenf.parameters() if p.requires_grad)
    results = {"AUC": test_auc, "Log-Density": np.mean(loss_test), "Inference Time" : np.mean(inference_times), "Parameters": num_params}
    save_json(results, f'{log_path}/test_results')
import os
import torch
import torch.optim as optim
import argparse
from dataloader import *
from model.model import *
import timeit
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils import *




import subprocess


def get_process_gpu_memory():
    """
    Get GPU memory used by CURRENT process only
    """
    pid = os.getpid()
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        process_memory = 0
        # Parse output
        for line in result.stdout.strip().split('\n'):
            if line:
                process_pid, memory = line.split(',')
                process_pid = int(process_pid.strip())
                memory = int(memory.strip())
                
                if process_pid == pid:
                    process_memory += memory
        
        return process_memory
        
    except Exception as e:
        print(f"Error getting process GPU memory: {e}")
        return None


def get_args():

    parser = argparse.ArgumentParser(description="Anomaly Detection")
    parser.add_argument("--checkpt", type=str, default='checkpoint', help="checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--log", type=int, default=20, help="How often to log model")
    parser.add_argument("--log_output", type=str, default="log")
    parser.add_argument("--name", type=str, default="WaveletNF")
    parser.add_argument("--gpu", action="store_true")
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
    parser.add_argument("--clean_training", action="store_true")
    # training parameters
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride_size", type=int, default=10)
    args = parser.parse_args()
    return args



def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def plot_histogram(log_likelihoods, labels, dataset):

    
    log_likelihoods = -1 * log_likelihoods
    #
   

    normalized_ll = sigmoid(log_likelihoods)
    # Separate by class
    class1_mask = labels == 0
    class2_mask = labels == 1

    class1_data = normalized_ll[class1_mask]
    class2_data = normalized_ll[class2_mask]

    # Create histograms
    n_bins = 10
    bins = np.linspace(0, 1.0, n_bins)  # Bins from 0 to 1
    

    class1_hist, bin_edges = np.histogram(class1_data, bins=bins)
    class2_hist, _ = np.histogram(class2_data, bins=bins)

    # Create bin labels
    bin_labels = [f'{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges)-1)]

    # Save to CSV
    df = pd.DataFrame({
        'bin': bin_labels,
        'class1': class1_hist,
        'class2': class2_hist
    })

    df.to_csv('figures/pmu_histogram_data.csv', index=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histograms side by side
    x = np.arange(len(bin_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, class1_hist, width, label='Normal', alpha=0.8)
    bars2 = ax.bar(x + width/2, class2_hist, width, label='Anomaly', alpha=0.8)

    ax.set_xlabel('Normalized Log-Likelihood Bins')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Normalized Log-Likelihoods by Class')
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/pmu_histogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    



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

    contaminated = not args.clean_training
    train_loader, val_loader, test_loader, n_sensors = load_data(dataset=args.dataset, contaminated = contaminated,
                                                                 window_size=args.window_size, stride=args.stride_size, batch_size=args.batch_size)
    k = int(args.k * n_sensors)
    # train model

    # create instance of CANF model
    wavenf = WaveletEnhancedNF(num_blocks=args.num_blocks,
                               hidden_d = args.st_units, 
                               wavelet_type = args.wavelet_type,
                               k = k,
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

    loss_val = []
    labels_val = []
    if args.example == 0:
        with torch.no_grad():
                wavenf.eval()
                for x, labels,  in val_loader:
                    x = x.to(device)
                    loss =  -1 * wavenf(x, take_mean=False).cpu().numpy()
                    #if not(args.attention == 0 or args.lam == 0.0):
                        #loss += l1_loss(wavenf, lam=args.lam).cpu().numpy()
                    #print(loss)
                    loss_val.append(loss)
                    labels_val.append(labels)
                
                #print(loss_val)
                loss_val = np.concatenate(loss_val)
                labels_val = np.concatenate(labels_val)

            # get threshold based on training data
                print("searching for threshold")
                threshold = search_opt_threshold(loss_val, labels_val)



    #print(test_loader.dataset.labels)
    with torch.no_grad():
        wavenf.eval()
        if args.dataset != "PMU":
            j = 0
            for x, labels  in test_loader:
                x = x.to(device)
                if args.example == 1:
                    loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                    #loss = loss.reshape(-1)
                    print("loss shape:", loss.shape)
                    loss_test.append(loss)
                elif args.example == 2:
                    loss = -1 * wavenf(x, take_mean=False, take_t_mean=False).cpu().numpy()
                    loss = loss.reshape(-1)
                    loss_test.append(loss)
                else:
                    start = timeit.default_timer()
                    loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                    #if not(args.attention == 0 or args.lam == 0):
                        #loss += l1_loss(wavenf, args.lam).cpu().numpy()
                    end = timeit.default_timer()
                    if j == 0:
                        torch.cuda.synchronize()
                        captured_memory = get_process_gpu_memory()
                        #print("captured memory:", captured_memory)
                        
                        # Also get PyTorch stats for comparison
                        pytorch_mem = torch.cuda.memory_allocated() / (1024**2)
                    j +=1
                    inference_times.append(end - start)
                    loss_test.append(loss)
                test_labels.append(labels)
                idx = test_loader.dataset.idx[i]

                i = i+1
        else:
            j = 0
            for loader in test_loader:
                for x, labels,  in loader:
                        x = x.to(device)
                        if args.example == 2:
                            loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                        else:   
                            start = timeit.default_timer()
                            loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                            #print(loss.shape)
                            #if not(args.attention == 0 or args.lam == 0.0):
                                #loss += l1_loss(wavenf, lam=args.lam).cpu().numpy()
                            end = timeit.default_timer()
                            if j == 0:
                                torch.cuda.synchronize()
                                captured_memory = get_process_gpu_memory()
                                #print("captured memory:", captured_memory)
                                
                                # Also get PyTorch stats for comparison
                                pytorch_mem = torch.cuda.memory_allocated() / (1024**2)
                            j +=1
                            inference_times.append(end - start)
                        #print(l1.shape)
                        loss_test.append(loss)
                        test_labels.append(labels)
                    
                
                #if args.example == 2:
                    #loss_test = np.concatenate(loss_test)
                    #print(loss_test)
                    #break



        



    # PMU Likelihoods Histogram
    if args.example == 1:
        print(f"Plotting {args.dataset} Likelihoods")
        plot_histogram(np.concatenate(loss_test), np.concatenate(test_labels), args.dataset)

    # WADI example, store likelihoods for further evaluation in notebook
    elif args.example == 2:
        #print(f"Plotting Wadi likelihoods")
        labels = test_loader.dataset.labels
        #print(labels)
        #print(np.concatenate(loss_test))
        wadi_results = {}
        wadi_results["likelihoods"] = np.concatenate(loss_test)
        wadi_results["labels"] = labels
        save_json(wadi_results, f'figures/wadi_likelihoods')


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
        results = get_metrics(loss_test, test_labels, threshold)
        num_params = sum(p.numel() for p in wavenf.parameters() if p.requires_grad)
        results["Log-Density"]= np.mean(loss_test)
        results["Inference_Time"] = np.mean(inference_times)
        results["GPU_Memory"] = captured_memory
        results["Model_Memory"] =  (sum(p.numel() for p in wavenf.parameters() if p.requires_grad))/ (1024 ** 2)
        results["Parameters"] = num_params
        save_json(results, f'{log_path}/test_results')
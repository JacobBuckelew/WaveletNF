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
    parser.add_argument("--checkpt", type=str, default='../checkpoint', help="checkpoint")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=float, default=0.05)
    parser.add_argument("--log", type=int, default=20, help="How often to log model")
    parser.add_argument("--log_output", type=str, default="../log")
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
    parser.add_argument("--lam", type=float, default=0.5)
    args = parser.parse_args()
    return args




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

    
num_features_range = [100, 200, 400, 800, 1600, 3200, 6400, 10000]
results = []
window_size = 64
memory_values = []


for num_feat in num_features_range:
    try:
        torch.cuda.empty_cache()
        
        k = int(args.k * num_feat)
    # train model

        # create instance of CANF model
        model = WaveletEnhancedNF(num_blocks=1,
                                hidden_d = 32, 
                                wavelet_type = args.wavelet_type,
                                k = k,
                                N = window_size,
                                n_heads = args.heads,
                                num_features = 1, 
                                st_units = 32, 
                                st_layers = 1,
                                num_entities=num_feat,
                                wavelet=args.wavelet,
                                attention=args.attention,
                                b_norm= True, 
                                momentum=0.95)
        model = model.to(device)
        
        
        # Input data
        batch_size = 64
        x = torch.randn(batch_size, window_size, num_feat).cuda()
        
        # Measure memory
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = model(x)
        
        torch.cuda.synchronize()
        
        memory = torch.cuda.max_memory_allocated() / (1024**2)
        
        
        results.append({
            'K': num_feat,
            'Memory_MiB': memory,
            'Status': 'Success'
        })
        memory_values.append(memory * 1.04858)
        
        #print(f"K={num_feat:4d}: Total {memory:7.1f} MiB")
        
        if num_feat > num_features_range[0]:
            prev_mem = results[-2]['Memory_MiB']
            factor = memory / prev_mem
            #print(f"         Scaling factor: {factor:.2f}× (expect ~4× for quadratic)")
        
        del model, x
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            results.append({
                'K': num_feat,
                'Memory_MiB': np.nan,
                'Status': 'OOM'
            })
            memory_values.append("OOM")
            torch.cuda.empty_cache()
        else:
            raise e
        
#print(memory_values)

data = list(zip(num_features_range, memory_values))
df = pd.DataFrame(data, columns=['D', 'Memory'])
df.to_csv(f"figures/WENFLOW_memory_values_{args.k}.csv")
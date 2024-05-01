#%%
import os
import argparse
import torch
import json
from utils import *
from models.GANF import GANF
import timeit
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# from data import fetch_dataloaders


parser = argparse.ArgumentParser()
# files
parser.add_argument('--data_dir', type=str, 
                    default='./data/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='/home/enyandai/code/checkpoint/model')
parser.add_argument('--name',default='GANF')
parser.add_argument('--dataset', type=str, default='swat')
# restore
parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='None')
parser.add_argument('--seed', type=int, default=10, help='Random seed to use.')
# made parameters
parser.add_argument('--n_blocks', type=int, default=6, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--batch_norm', type=bool, default=False)
# training params
parser.add_argument('--batch_size', type=int, default=512)

args = parser.parse_known_args()[0]
args.cuda = torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


print(args)
import random
import numpy as np
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


from dataset import *
infer_times = []

if args.dataset == "swat":
    train_loader, val_loader, test_loader, n_sensor = load_swat(args.batch_size)

elif args.dataset == "psm":
    print("Loading PSM")
    train_loader, val_loader, test_loader, n_sensor = load_psm(args.batch_size, window_size=60,
                                                      stride_size=10, train_split=0.60)


model = GANF(args.n_blocks, 1, args.hidden_size, args.n_hidden, dropout=0.0, batch_norm=args.batch_norm)
model = model.to(device)
os.chdir("../baselines/GANF/")

print(os.getcwd())
model.load_state_dict(torch.load(f"./checkpoint/{args.name}/{args.name}_best.pt"))
A = torch.load(f"./checkpoint/{args.name}/graph_best.pt").to(device)
model.eval()
log_path = "./log/test_results/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
#%%
densities = []
loss_test = []
with torch.no_grad():
    labels = []
    for x, _, _ in test_loader:
        x = x.to(device)
        start = timeit.default_timer()
        loss = -model.test(x, A.data).cpu().numpy()
        end = timeit.default_timer()
        infer_times.append(end - start)
        loss_test.append(loss)
    label = np.asarray(test_loader.dataset.label, dtype=int)
    labels.append(label)

labels = np.concatenate(labels)
loss_test = np.concatenate(loss_test)


# generate final results

results = evaluate(labels, loss_test)
results["Parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
results["Inference Time"] = np.mean(infer_times)
save_json(results, f"{log_path}/{args.name}")
# %%

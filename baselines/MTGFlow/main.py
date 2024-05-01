#%%
import os
import argparse
import torch
from utils import *
import timeit
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='Data/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset')
parser.add_argument("--dataset", type=str, default="SWAT")

parser.add_argument('--graph', type=str, default='None')
parser.add_argument('--model', type=str, default='MAF')
parser.add_argument('--seed', type=int, default=7)


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)
parser.add_argument('--gpu', type=int, default=3)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



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
#%%
print("Loading dataset")

from Dataset import loader_SWat, loader_PSM

if args.dataset == 'SWAT':
    train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.dataset == 'PSM':
    train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.batch_size, args.window_size, args.stride_size, args.train_split)


print(os.getcwd())
os.chdir("../baselines/MTGFlow/")
#%%
model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
model = model.to(device)

    #%%
from torch.nn.utils import clip_grad_value_
import matplotlib.pyplot as plt
save_path = os.path.join(args.output_dir,args.name)
if not os.path.exists(save_path):
    os.makedirs(save_path)

log_path = "./log/results/"
if not os.path.exists(log_path):
    os.makedirs(log_path)

print(save_path)
loss_best = 1000
  
lr = args.lr 
optimizer = torch.optim.Adam([
    {'params':model.parameters(), 'weight_decay':args.weight_decay},
    ], lr=lr, weight_decay=0.0)

start = timeit.default_timer()
for epoch in range(40):
    print(epoch)
    loss_train = []

    model.train()
    for x,_,idx in train_loader:
        x = x.to(device)

        optimizer.zero_grad()
        loss = -model(x,)

        total_loss = loss

        total_loss.backward()
        clip_grad_value_(model.parameters(), 1)
        optimizer.step()
        loss_train.append(loss.item())



    loss_val = []
    model.eval()
    with torch.no_grad():
        for x, _, idx in val_loader:
            x = x.to(device)
            loss = -model.test(x, ).cpu().numpy()
            loss_val.append(loss)

    loss_val = np.concatenate(loss_val)
    
    if np.mean(loss_val) < loss_best:
        print("saving model")
        loss_best = np.mean(loss_val)
        torch.save({
        'model': model.state_dict(),
        }, f"{save_path}/model.pth")


end = timeit.default_timer()
results = {}
results["train_time"] = end - start
save_json(results, f"{log_path}/{args.name}")

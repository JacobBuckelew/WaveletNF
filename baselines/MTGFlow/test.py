#%%
import os
import argparse
import torch
import timeit
from utils import *
from models.MTGFLOW import MTGFLOW
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve 

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, 
                    default='../u_s/input/SWaT_Dataset_Attack_v0.csv', help='Location of datasets.')
parser.add_argument('--output_dir', type=str, 
                    default='./checkpoint/')
parser.add_argument('--name',default='SWaT', help='the name of dataset')

parser.add_argument('--model', type=str, default='MAF')
parser.add_argument("--dataset", type=str, default="SWAT")


parser.add_argument('--n_blocks', type=int, default=1, help='Number of blocks to stack in a model (MADE in MAF; Coupling+BN in RealNVP).')
parser.add_argument('--n_components', type=int, default=1, help='Number of Gaussian clusters for mixture of gaussians models.')
parser.add_argument('--hidden_size', type=int, default=32, help='Hidden layer size for MADE (and each MADE block in an MAF).')
parser.add_argument('--n_hidden', type=int, default=1, help='Number of hidden layers in each MADE.')
parser.add_argument('--input_size', type=int, default=1)
parser.add_argument('--batch_norm', type=bool, default=False)
parser.add_argument('--train_split', type=float, default=0.6)
parser.add_argument('--stride_size', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--window_size', type=int, default=60)
parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate.')



args = parser.parse_known_args()[0]
#args.cuda = torch.cuda.is_available()
#device = torch.device("cuda" if args.cuda else "cpu")
device = "cpu"
save_path = os.path.join(args.output_dir,args.name)

from Dataset import loader_SWat, loader_PSM

if args.dataset == 'SWAT':
    train_loader, val_loader, test_loader, n_sensor = loader_SWat(args.batch_size, args.window_size, args.stride_size, args.train_split)

elif args.dataset == 'PSM':
    train_loader, val_loader, test_loader, n_sensor = loader_PSM(args.batch_size, args.window_size, args.stride_size, args.train_split)


os.chdir("../baselines/MTGFlow/")
infer_times = []
#%%
model = MTGFLOW(args.n_blocks, args.input_size, args.hidden_size, args.n_hidden, args.window_size, n_sensor, dropout=0.0, model = args.model, batch_norm=args.batch_norm)
model = model.to(device)

checkpoint = torch.load(f"{save_path}/model.pth")
model.load_state_dict(checkpoint['model'])

log_path = "./log/test_results/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
model.eval()

loss_test = []
with torch.no_grad():
    if args.dataset != "RTDS":
        for x,_,idx in test_loader:

            x = x.to(device)
            start = timeit.default_timer()
            loss = -model.test(x, ).cpu().numpy()
            end = timeit.default_timer()
            infer_times.append(end - start)
            loss_test.append(loss)
        loss_test = np.concatenate(loss_test)
#loss_test = np.concatenate(loss_test)
if args.dataset != "RTDS":
    labels = np.asarray(test_loader.dataset.label,dtype=int)
results = evaluate(labels, loss_test)
print(results["AUC"])
results["Parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
results["Inference Time"] = np.mean(infer_times)
save_json(results, f"{log_path}/{args.name}")
import os
os.chdir("../")
import torch
import torch.optim as optim
import argparse
from dataloader import *
from model.model import *
import time
from sklearn.metrics import roc_auc_score
from utils import *


def l1_loss(model, lam):
    loss = 0
    for i in range(len(model.flow)):
        loss += torch.norm(model.flow[i].encoder.attention.head.weight, p=1)

    #print("loss:", loss)
    return lam * loss

def get_args():

    parser = argparse.ArgumentParser(description="OOD Detection")
    parser.add_argument("--checkpt", type=str, default='', help="checkpoint")
    parser.add_argument("--log_output", type=str, default="log")
    parser.add_argument("--model", type=str, default="CANF")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpt_output", type=str, default="checkpoint")
    parser.add_argument("--dataset", type=str, default="SWAT")
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=1)
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--wavelet_type", type=str, default="haar")
    parser.add_argument("--b_norm", type=bool, default=True)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--stride_size", type=float, default=10)
    args = parser.parse_args()
    return args


# MAIN

if __name__ == "__main__":
    # get args
    args = get_args()

    config_path = "../configs/"
    output_path = f"../configs/{args.dataset}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    best_cfg = -1
    for seed in [args.seed]:
        best_val_cfg = 0
        best_val_loss = np.inf
        for j in range(50):
            # get configs
            np.random.seed(seed)
            torch.manual_seed(seed)
            if args.gpu != -1:
                torch.cuda.manual_seed_all(seed)
                cuda = torch.cuda.is_available()
                device = torch.device(f"cuda:{args.gpu}")
            else:
                device = torch.device("cuda:cpu")
            model_config = load_json(config_path + f"cfg_{j}")
            configs = dotdict(model_config)
            print(configs)
            if args.dataset == "SWAT":
                train_loader, val_loader, test_loader, n_sensor = load_data("SWAT", args.window_size, args.stride_size, configs.batch_size)
            elif args.dataset == "WADI":
                train_loader, val_loader, test_loader, n_sensor =load_data("WADI", args.window_size, args.stride_size, configs.batch_size)
            elif args.dataset.startswith('machine'):
                train_loader, val_loader, test_loader, n_sensor= load_data(args.dataset, args.window_size, args.stride_size, configs.batch_size)
            elif args.dataset == "PSM":
                train_loader, val_loader, test_loader, n_sensor = load_data("PSM", args.window_size, args.stride_size, configs.batch_size)
            elif args.dataset == "RTDS":
                train_loader, val_loader, test_loaders, n_sensor = load_data("RTDS", args.window_size, args.stride_size, configs.batch_size)

            wavenf = WaveletEnhancedNF(num_blocks=configs.num_blocks,
                               hidden_d = configs.st_units, 
                               wavelet_type = args.wavelet_type,
                               N = args.N,
                               n_heads = args.heads,
                               num_features = 1, 
                               st_units = configs.st_units, 
                               st_layers = configs.st_layers,
                               num_entities=n_sensor,
                               b_norm= True, 
                               momentum=0.95)
            
            
            wavenf.to(device)
        
            optimizer = optim.Adam(
            [{'params': wavenf.parameters(), 'weight_decay': args.wdecay}],
            lr=configs.lr, weight_decay=0.0
            )

            # Learning rate scheduler
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)
            # Training Loop, only use train and validation loaders here
            best_loss = np.inf
            for epoch in range(configs.epochs):
                loss_train = []
                wavenf.train()
                for x, _, in train_loader:
                    x = x.to(device)
                    # zero gradients
                    optimizer.zero_grad()
                    # pass data through the model
                    loss = -1* wavenf(x)
                    loss += l1_loss(wavenf, lam=configs.lam)
                    # backward step
                    total_loss = loss
                    loss.backward()
                    optimizer.step()
                    #print(total_loss.shape)
                    loss_train.append(total_loss.item())

                scheduler.step()
            
                # validate on calibration data
                wavenf.eval()
                loss_val = []
                with torch.no_grad():
                    for x, _,  in val_loader:
                        x = x.to(device)
                        loss =  -1 * wavenf(x).cpu().numpy()
                        #print(loss)
                        loss_val.append(loss)
                
                t_loss = np.mean(loss_train) + l1_loss(wavenf, lam=configs.lam)
                v_loss = (np.mean(loss_val) + l1_loss(wavenf, lam=configs.lam)).cpu().item()
                print("=====================================")
                print(f"Epoch {epoch}: train loss = {t_loss}, val loss = {v_loss}")
                # checkpoint for saving best params

                if v_loss < best_loss:
                    best_loss = v_loss
            #results = {"final train_loss": np.mean(loss_train), "final val_loss": np.mean(loss_val), "best val_loss:" : best_val_loss, "best epoch": best_epoch}
            # update overall best validation loss across all hyperparameters
                if best_loss < best_val_loss:
                    best_val_loss = best_loss
                    best_val_cfg = j 
                    print(best_val_loss)
                    print(f"saving config {j}")
                    results = {"best_val_loss": best_val_loss, "best_cfg": best_val_cfg}
                    save_json(results, f'{output_path}/opt_cfg')

        
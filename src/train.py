import os
os.chdir("../")
import torch
import torch.optim as optim
import argparse
from dataloader import *
from model.model import *
import time
from sklearn.metrics import roc_auc_score
from utils import save_json

def l1_loss(model, lam):
    loss = 0
    for i in range(len(model.flow)):
        loss += torch.norm(model.flow[i].encoder.attention.W_o.weight, p=1)

    #print("loss:", loss)
    return lam * loss

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
    parser.add_argument("--b_norm", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wavelet_type", type=str, default="haar")
    parser.add_argument("--wavelet", type=int, default=1)
    parser.add_argument("--attention", type=int, default=1)
    parser.add_argument("--wdecay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    # training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lam", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--stride_size", type=int, default=10)
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
    print("training on ",device)


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
    print("on device")
    time.sleep(20)
    wavenf.train()

    # save checkpt path
    checkpt_path = os.path.join(args.checkpt, args.name)
    print("Saving model parameters to: ",checkpt_path)
    if not os.path.exists(checkpt_path):
        os.makedirs(checkpt_path)

    log_path = os.path.join(args.log_output, args.name)
    print("Logging final train/val loss to: ", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
        
    optimizer = optim.Adam(
        [{'params': wavenf.parameters(), 'weight_decay': args.wdecay}],
        lr=args.lr, weight_decay=0.0
    )

    if args.dataset == "SWAT":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.50)
    # Training Loop, only use train and validation loaders here
    best_val_loss = 100000
    best_epoch = 0
    start_time = time.time()
    for epoch in range(args.epochs):
        loss_train = []
        wavenf.train()
        for x, _, in train_loader:
            x = x.to(device)
            # zero gradients
            optimizer.zero_grad()
            # pass data through the model
            loss = -1* wavenf(x)
            if not(args.attention == 0 or args.lam == 0.0):
                loss += l1_loss(wavenf, lam=args.lam)
            # backward step
            total_loss = loss
            loss.backward()
            optimizer.step()
            #print(total_loss.shape)
            loss_train.append(total_loss.item())

        scheduler.step()
        
        # validate on calibration data
        #wavenf.eval()
        loss_val = []
        labels = []
        loss_test = []
        test_labels = []
        wavenf.eval()
        with torch.no_grad():
            for x, _,  in val_loader:
                x = x.to(device)
                loss =  -1 * wavenf(x).cpu().numpy()
                if not(args.attention == 0 or args.lam == 0.0):
                    loss += l1_loss(wavenf, lam=args.lam).cpu().numpy()
                #print(loss)
                loss_val.append(loss)
            
            for x, labels,  in test_loader:
                x = x.to(device)
                loss = -1 * wavenf(x, take_mean=False).cpu().numpy()
                #print(loss.shape)
                if not(args.attention == 0 or args.lam == 0.0):
                    loss += l1_loss(wavenf, lam=args.lam).cpu().numpy()
                
                #print(l1.shape)
                loss_test.append(loss)
                test_labels.append(labels)
            
        loss_test = np.concatenate(loss_test)
        test_labels = np.concatenate(test_labels)
        test_auc = roc_auc_score(test_labels, loss_test)
        print("=====================================")
        print(f"Epoch {epoch}: train loss = {np.mean(loss_train)}, val loss = {np.mean(loss_val)}, test auc = {test_auc}")
        # checkpoint for saving best params
        if np.mean(loss_val) < best_val_loss:
            best_val_loss = np.mean(loss_val)
            torch.save(wavenf.state_dict(), os.path.join(checkpt_path, "best_params.pt"))
            best_epoch = epoch
            print("saving best params")
    end_time = time.time()
    train_time = end_time - start_time
    results = {"final train_loss": np.mean(loss_train), "final val_loss": np.mean(loss_val), "best val_loss:" : best_val_loss, "best epoch": best_epoch, "train_time": train_time}
    save_json(results, f'{log_path}/results')
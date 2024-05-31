import os, sys
#sys.path.insert(1, '../src')
from utils import *

# generate cfgs for hyperparameter tuning:
directory = "../configs/"

if __name__ == "__main__":
    # save 50 configurations for each window size
    np.random.seed(7)
    for s in range(50):
        configs = {

            'lr' : np.random.choice([1e-3, 2e-3, 1e-4, 5e-4]),
            'epochs' : 30,
            'batch_size' : np.random.choice([512]),
            'num_blocks' : np.random.choice([1,2]),
            'st_units' : np.random.choice([8, 16, 32, 64]),
            'st_layers' : 1,
            'lam' : np.random.choice(list(np.linspace(0.1, 1.0,num=9, endpoint=False)))
        }

        save_json(configs, f'{directory}cfg_{s}')
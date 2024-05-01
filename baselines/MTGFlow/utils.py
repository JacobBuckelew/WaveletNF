import numpy as np
import json
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NpEncoder, indent=4)

def load_json(file):
    with open(file) as f:
        obj = json.load(f)
    return obj

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def evaluate(labels, losses):

    results = {}
    auc = roc_auc_score(labels, losses)
  
    results["AUC"] = auc
    results["Log-Density"] = np.mean(losses)
    

    return results




def ROC(args, y_test,y_pred):
    auc=roc_auc_score(y_test,y_pred)
    print('auroc', auc)
    return auc
class log():
    def __init__(self) -> None:
        self.roc_auc_max = 0
        self.f1_max = 0
    def print_result(self, y_test, y_pred, model, i, args):
        y_test = np.nan_to_num(y_test)
        y_pred = np.nan_to_num(y_pred)

        auc=roc_auc_score(y_test,y_pred)
      
        if not os.path.exists("./{}{}modelroc".format(args.model, args.name,i, auc)):
            os.makedirs("./{}{}modelroc/".format(args.model,args.name, i, auc))
  
        if self.roc_auc_max < auc:
            self.roc_auc_max = auc
       
          
        print('auroc:{:.4f}'.format(auc))
        

import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, roc_curve, recall_score, f1_score, confusion_matrix, auc

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def load_json(file):
    with open(file) as f:
        obj = json.load(f)
    return obj


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, cls=NpEncoder, indent=4)


# Get F1, Precision, Recall, and AUC

def get_metrics(scores, y_true):

    auc = roc_auc_score(y_true, scores)
    threshold = search_opt_threshold(scores, y_true)
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    epsilon = 1e-5
    prec = tp/ (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = (2 * prec * recall)/ (prec + recall + epsilon)

    #prec = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    fpr, tpr, thr = roc_curve(y_true, scores)

    results = {}
    results["AUC"] = auc
    results["Precision"] = prec
    results["Recall"] = recall
    results["F1"] = f1
    results["FPR"] = fpr
    results["TPR"] = tpr

    return results

    
def search_opt_threshold(scores, y_true, best='f1'):
    """
    Grid search the optimal threshold for flagging anomalies on the testing dataset
    @param scores: anomaly scores (NLL)
    @param y_true: real binary classes
    @return (optimal F1 score, optimal threshold)
    """
    opt_thrd = np.quantile(scores, 0.95).astype(float)  # default threshold
    opt = -np.inf
    for thr in scores:
        y_pred = scores >= thr
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            if tp + fp == 0 or tp + fn == 0:
                continue
        except:
            continue
        epsilon = 1e-5
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * recall * precision / (recall + precision + epsilon)
        fpr, tpr, threshold = roc_curve(y_true, y_pred)
        auc_score = auc(fpr, tpr)
        if best == 'f1':
            if f1 > opt:
                opt_thrd = thr
                opt = f1
        elif best == 'auc':
            if auc_score > opt:
                opt_thrd = thr
                opt = auc_score
    return float(opt_thrd)



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

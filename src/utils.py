import json
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, roc_curve, recall_score, f1_score, confusion_matrix, auc
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.metrics import accuracy_score
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


# Adjustment strategy used in AnomalyTransformer (see https://github.com/thuml/Anomaly-Transformer/blob/main/solver.py)

def adjust_detection(pred, gt):

    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

        pred = np.array(pred)
        gt = np.array(gt)
    
    return pred, gt



# Get F1, Precision, Recall, and AUC

def get_metrics(scores, y_true, threshold):

    #print("threshold:", threshold)
    #print("scores:", scores)
    #print("y_true:", y_true)
    #print("median:", median)
    #print("threshold:", threshold)
    # calculate auc score
    y_pred = (scores > threshold).astype(int)
    auc_ = roc_auc_score(y_true, scores)
    #threshold = search_opt_threshold(scores, y_true)

    pred, gt = adjust_detection(y_pred.tolist(), y_true.tolist())

    # get final metrics
    precision_, recall_, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
    
    
    
    
    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    #epsilon = 1e-5
    #prec = tp/ (tp + fp + epsilon)
    #recall = tp / (tp + fn + epsilon)
    #f1 = (2 * prec * recall)/ (prec + recall + epsilon)

    #prec = precision_score(y_true, y_pred)
    #recall = recall_score(y_true, y_pred)
    #f1 = f1_score(y_true, y_pred)
    fpr, tpr, thr = roc_curve(y_true, scores)



    #precision, recall, thresholds = precision_recall_curve(y_true, scores)
    #auprc = auc(recall, precision)
    # Find precision at 90% recall
    #target_recall = 0.90
    #idx = np.argmin(np.abs(recall - target_recall))

    #precision_at_90_recall = precision[idx]
    #actual_recall = recall[idx]

    #target_precision = 0.90
    #idx = np.argmin(np.abs(precision - target_precision))

    #recall_at_90_precision = recall[idx]
    #actual_precision = precision[idx]

    results = {}
    results["AUC"] = auc_
    results["Precision"] = precision_
    results["Recall"] = recall_
    results["F1"] = f_score
    results["FPR"] = fpr
    results["TPR"] = tpr
    #results["AUPRC"] = auprc
    #results["PRECISION@"] = precision_at_90_recall
    #results["RECALL@"] = recall_at_90_precision

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
        if best == 'f1':
            if f1 > opt:
                opt_thrd = thr
                opt = f1
    return float(opt_thrd)


def get_threshold(scores, labels):

    #q0 = np.median(scores)
    #print("q0:", q0)
    #deviations = np.abs(scores - q0)
    #k = 6.5
    #med_deviations = np.median(deviations)

    #q1 = med_deviations * k

    #q1 = np.percentile(scores, 25)
    #print("q1:", q1)
    #if dataset.startswith('machine'):
       # q = np.percentile(scores, 90)
    #elif dataset == "PMU":
        #q = np.percentile(scores, 85)

    threshold = search_opt_threshold(scores, labels)
    
    #q3 = np.percentile(scores, 75)
    #threshold = q3 + (1.5 * (q3 - q1))

    #print("mean:", np.mean(scores))

    #mean = np.mean(scores)

    #std = np.std(scores)

    #print("q1:", q1)

    return threshold


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

# %%
from utils import *
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

results = dotdict({})
for model in ["WaveletNF", "WENFW", "WENFA", "RealNVP", "coif1", "coif2", "db1", "db2", "haar", "k0.05", "k0.10", "k0.15", "k0.20", "k0.25", "window16", "window32", "window48","window64", "window96"]:

    model_results = dotdict({})

    for dataset in ["PMU", "wadi","swat"]:
        model_results[dataset] = dotdict({"AUC": 0, "AUC_std": 0})

        auc_values = []
        prec_values = []
        aucpr_values = []
        auc = 0
        recall_values = []
        f1_values = []
        infer_time_values = []
        parameter_values = 0
        train_time = []
        densities = []
        if model in ["k0.05", "k0.10", "k0.15", "k0.20", "k0.25", "db1", "db2", "coif1", "coif2", "haar", "window16", "window32", "window48","window64", "window96"]:
            seeds = [6, 7, 8]
        else:
            seeds = [6, 7, 8, 9, 10]
        for seed in seeds:
            log = load_json(f"../log/{model}_{dataset}_seed_{seed}/test_results")
            log = dotdict(log)
            auc_values.append(log["AUC"])
            if log["AUC"] > auc:
                auc = log["AUC"]
            tpr = log["TPR"]
            fpr = log["FPR"]
            if model == "WaveletNF":
                if dataset == "swat" and seed ==7 :
                    df = pd.DataFrame([fpr, tpr]).T.round(4)
                    df.columns = [["x", "y"]]
                    df.to_csv(f"ROC_{model}_{dataset}.csv", index=False)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"{model} on {dataset} (AUC={log['AUC']:.4f})")
                    plt.title("ROC Curve for WaveletNF on SWaT Dataset")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.savefig("../figures/ROC_WaveletNF_swat.png", dpi=300, bbox_inches='tight')
                    print("Saved SWaT ROC Curve to ../figures/ROC_WaveletNF_swat.png")
                    
                if dataset == "wadi" and seed == 6:
                    df = pd.DataFrame([fpr, tpr]).T.round(4)
                    df.columns = [["x", "y"]]
                    df.to_csv(f"ROC_{model}_{dataset}.csv", index=False)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"{model} on {dataset} (AUC={log['AUC']:.4f})")
                    plt.title("ROC Curve for WaveletNF on WADI Dataset")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.savefig("../figures/ROC_WaveletNF_wadi.png", dpi=300, bbox_inches='tight')
                    print("Saved WADI ROC Curve to ../figures/ROC_WaveletNF_wadi.png")

                if dataset == "PMU" and seed == 6:
                    df = pd.DataFrame([fpr, tpr]).T.round(4)
                    df.columns = [["x", "y"]]
                    df.to_csv(f"ROC_{model}_{dataset}.csv", index=False)
                    fig = plt.figure()
                    plt.plot(fpr, tpr, label=f"{model} on {dataset} (AUC={log['AUC']:.4f})")
                    plt.title("ROC Curve for WaveletNF on PMU Dataset")
                    plt.xlabel("False Positive Rate")
                    plt.ylabel("True Positive Rate")
                    plt.savefig("../figures/ROC_WaveletNF_PMU.png", dpi=300, bbox_inches='tight')
                    print("Saved PMU ROC Curve to ../figures/ROC_WaveletNF_PMU.png")

            infer_time_values.append(log["Inference_Time"])
            densities.append(log["Log-Density"])
            parameter_values = log["Parameters"]
            prec_values.append(log["Precision"])
            recall_values.append(log["Recall"])
            f1_values.append(log["F1"])


            log = load_json(f"../log/{model}_{dataset}_seed_{seed}/results")
            log = dotdict(log)
            train_time.append(log["train_time"])
            

        
        model_results[dataset].AUC = np.mean(auc_values)
        model_results[dataset].AUC_std = np.std(auc_values)
        model_results[dataset].Precision = np.mean(prec_values)
        model_results[dataset].Precision_std = np.std(prec_values)
        model_results[dataset].Recall = np.mean(recall_values)
        model_results[dataset].Recall_std = np.std(recall_values)
        model_results[dataset].F1 = np.mean(f1_values)
        model_results[dataset].F1_std = np.std(f1_values)

        model_results[dataset].Parameters = parameter_values
        model_results[dataset].Time = np.mean(infer_time_values)
        model_results[dataset].Train_Time = np.mean(train_time)
        model_results[dataset].Train_Time_std = np.std(train_time)
        model_results[dataset].Time_std = np.std(infer_time_values)
    
    results[model] = model_results


# %%
print("WENFlow Detection and Efficiency Results (Tables 4, 5):")
print("================================")
print("PMU:")
print(results.WaveletNF.PMU)

# %%
print("WADI:")
print(results.WaveletNF.wadi)

# %%
print("SWaT:")
print(results.WaveletNF.swat)

# %%
print("WENFlow Ablation Results (Table 6):")
print("================================")
print("WF\A PMU:")
print(results.WENFA.PMU)

print("WF\A WADI:")
print(results.WENFA.wadi)

print("WF\A SWaT:")
print(results.WENFA.swat)

print("WF\W PMU:")
print(results.WENFW.PMU)

print("WF\W WADI:")
print(results.WENFW.wadi)

print("WF\W SWaT:")
print(results.WENFW.swat)  

print("RealNVP PMU:")
print(results.RealNVP.PMU)

print("RealNVP WADI:")
print(results.RealNVP.wadi)

print("RealNVP SWaT:")
print(results.RealNVP.swat)

# %%



# %%
print("Sensitivity Analysis Figures (Figures 6, 7, 8):")
print("================================")

print("Sensitivity Analysis on K: saved as ../figures/sensitivity_k.png")

k_values = ["k0.05", "k0.10", "k0.15", "k0.20", "k0.25"]
datasets = ['PMU', 'wadi', 'swat']  

# Extract AUC values for each dataset
plt.figure(figsize=(10, 6))

for dataset in datasets:
    auc_values = []
    for k in k_values:
        k_key = f'{k}'  
        auc = results[k_key][dataset]['AUC']
        auc_values.append(auc)
    
    
    plt.plot(k_values, auc_values, marker='o', label=dataset, linewidth=2, markersize=8)

plt.xlabel('k value', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('AUC vs k value', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0) 
plt.tight_layout()
plt.savefig('../figures/sensitivity_k.png', dpi=300, bbox_inches='tight')
plt.show()



# %%
print("Sensitivity Analysis on Wavelet Type: saved as ../figures/sensitivity_wavelet.png")
wavelet_types = ["haar", "coif1", "coif2", "db1", "db2"]
datasets = ['PMU', 'wadi', 'swat']  

# Extract AUC values for each dataset
plt.figure(figsize=(10, 6))

for dataset in datasets:
    auc_values = []
    for wavelet in wavelet_types:
        wavelet_key = f'{wavelet}'  # Creates 'haar', 'coif1', etc.
        auc = results[wavelet_key][dataset]['AUC']
        auc_values.append(auc)
    
    plt.plot(wavelet_types, auc_values, marker='o', label=dataset, linewidth=2, markersize=8)

plt.xlabel('Wavelet Type', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('AUC vs Wavelet Type', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)  # AUC ranges from 0 to 1
plt.tight_layout()
plt.savefig('../figures/sensitivity_wavelet.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
print("Sensitivity Analysis on Window Size: saved as ../figures/sensitivity_window.png")

window_sizes = ["window16", "window32", "window48", "window64", "window96"]
datasets = ['PMU', 'wadi', 'swat']  

# Extract AUC values for each dataset
plt.figure(figsize=(10, 6))

for dataset in datasets:
    auc_values = []
    for window in window_sizes:
        window_key = f'{window}'  # Creates '16', '32', etc.
        auc = results[window_key][dataset]['AUC']
        auc_values.append(auc)
    
    plt.plot(window_sizes, auc_values, marker='o', label=dataset, linewidth=2, markersize=8)

plt.xlabel('Window Size', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.title('AUC vs Window Size', fontsize=14)
plt.legend(fontsize=10, loc='best')
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.0)  # AUC ranges from 0 to 1
plt.tight_layout()
plt.savefig('../figures/sensitivity_window.png', dpi=300, bbox_inches='tight')
plt.show()

# %%




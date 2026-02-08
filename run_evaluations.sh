#!/bin/bash
set -e

echo "=================================="
echo "Starting Evaluation Pipeline"
echo "=================================="

mkdir -p results
mkdir -p figures

echo ""
echo "Step 1: Running Evaluation Code for WaveletNF, ablation models, and sensitivity analyses... will take approximately 30-60 minutes"
echo "=================================="

bash src/runners/test_pmu.sh
bash src/runners/test_swat.sh
bash src/runners/test_wadi.sh

bash src/runners/test_no_attention_pmu.sh
bash src/runners/test_no_wavelet_pmu.sh
bash src/runners/test_realnvp_pmu.sh

bash src/runners/test_no_attention_swat.sh
bash src/runners/test_no_wavelet_swat.sh
bash src/runners/test_realnvp_swat.sh

bash src/runners/test_no_attention_wadi.sh
bash src/runners/test_no_wavelet_wadi.sh
bash src/runners/test_realnvp_wadi.sh

bash src/runners/test_k_pmu.sh
bash src/runners/test_k_swat.sh
bash src/runners/test_k_wadi.sh

bash src/runners/test_wavelet_pmu.sh
bash src/runners/test_wavelet_swat.sh
bash src/runners/test_wavelet_wadi.sh   

bash src/runners/test_window_pmu.sh
bash src/runners/test_window_swat.sh
bash src/runners/test_window_wadi.sh



echo ""
echo "Step 2: Running get_results.py... Table 4, 5 and 6, and Figures 6, 7, and 8 in the paper "
echo "=================================="
python src/get_results.py | tee results/results.txt

echo ""
echo "Step 3: Running Optional Scaling Analysis if GPU is Available.. Figure 5 in the paper."
echo "=================================="
bash src/runners/test_scaling.sh
python src/visualize_scaling.py

echo ""
echo "Step 4: Running PMU Histogram Example... Figure 3 in the paper"
echo "=================================="
bash src/runners/likelihoods_pmu.sh

echo ""
echo "Step 5: Running WADI Example... Figure 4 in the paper"
echo "=================================="
bash src/runners/wadi_example.sh
python src/visualize_wadi.py

echo ""
echo "Step 6: Running Cascade Example... Figure 9 in the paper"
echo "=================================="
python src/cascade_example.py

echo ""
echo "=================================="
echo "✓ Evaluation Complete!"
echo "=================================="
echo ""
echo "Numerical Results saved to: results/"
ls -lh results/
echo ""
echo "Figures saved to: figures/"
ls -lh figures/
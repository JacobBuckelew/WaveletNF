# Artifact Evaluation Documentation

## Paper and BibTeX Citation

Check out the link below and download the PDF of our full paper (including appendices):
https://scopelab.ai/pub-summaries/iccps2026_wenflow/ 


If you choose to cite our paper, please use the following BibTeX entry:

```bibtex
@inproceedings{iccps2026_wenflow,
  author = {Buckelew, Jacob and Talusan, Jose Paolo and Sivaramakrishnan, Vasavi and Mukhopadhyay, Ayan and Srivastava, Anurag and Dubey, Abhishek},
  title = {WENFlow: Scalable Attention for Unsupervised Spatiotemporal Anomaly Detection in High-Dimensional Cyber-Physical Systems},
  year = {2026},
  booktitle = {Proceedings of the HSCC/ICCPS 2026: 29th ACM International Conference on Hybrid Systems: Computation and Control and 17th ACM/IEEE International Conference on Cyber-Physical Systems},
  location = {Saint Malo, France},
  series = {HSCC/ICCPS '26}
}
```

## Requirements and Setup

### System Requirements

**Recommended Hardware:**
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- 16GB+ RAM
- 50GB+ available disk space

**Software:**
- Docker
- Ubuntu 20.04 or later (or compatible Linux distribution)
- Docker 19.03 or later
- NVIDIA drivers (version 450+)
- NVIDIA Container Toolkit


**Note:** The artifact can run on CPU-only systems, but evaluation will be significantly slower.


#### Step 1: Install NVIDIA Container Toolkit (For GPU Support)

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

Verify GPU access:
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
# Should display your GPU information


```
### Docker Image Details

- **Base Image:** `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04`
- **Python Version:** 3.11.5
- **PyTorch Version:** 2.0.0 with CUDA 11.8 support
- **Key Dependencies:** See `requirements.txt` in repository


#### Step 1: Access Code Repo

The code associated with the artifact can be found in WaveletNF/

```bash
# Enter code repo
cd WaveletNF
```

#### Step 2: Build Docker Image

```bash
# Build the Docker image (this may take a few minutes)
docker build -t my-evaluation:latest .
```

The build process will:
- Download base NVIDIA CUDA image
- Install Python 3.11.5 and dependencies
- Download the SWaT, WADI, and PMU datasets from Google Drive
- Set up the evaluation environment

#### Step 3: Create Output Directories

```bash
# Create local directories for results and figures if it doesn't already exist
mkdir -p results figures
```

---

## Expected Reproducibility Outcomes

### What to Expect

This artifact provides full reproducibility of the experimental results presented in the paper. Specifically:

1. **Quantitative Results**: All classification metrics (accuracy, F1-score, precision, recall, AUC) and efficiency metrics (Parameter counts, GPU memory, inference time) found in Tables 4, 5, and 6.

2. **Qualitative Results**: All figures and visualizations can be regenerated with the same visual patterns. Figures in the paper will appear slightly different from those generated in this evaluation since we will be using python's matplotlib library for simplicity. All plots in the paper are generated using PGFPlots in LaTeX to enhance their visual quality.

3. **Efficiency Results**: Some system-dependent metrics such as inference time may appear slightly different from the final results shown in the paper. This is expected since these metrics will depend on GPU model, CPU specifications, and system load.


---

## Usage Scenarios

### Scenario 1: Full Reproduction (Recommended)

Reproduce all experimental results from the paper with a single command.

**Command:**
```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/figures:/app/figures \
    my-evaluation:latest
```

**Expected outputs:**
- `results/results.txt`: All quantitative metrics
- `figures/*.png`: All figures from the paper


**Verification:**
```bash
# Check quantitative results after running the evaluation script
cat results/results.txt

# List generated figures after running the evaluation script
ls figures/


### Scenario 2: Partial Reproduction

Run specific experiments only:

**Step 1: Start interactive shell**
```bash
docker run --rm --gpus all -it \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/figures:/app/figures \
    my-evaluation:latest /bin/bash
```

**Step 2: Run specific scripts**
```bash
# Only run scaling analysis (Figure 5)
python src/test_scaling.py
python src/visualize_scaling.py

# Only run PMU histogram example (Figure 3)
bash src/runners/likelihoods_pmu.sh

# Only run WADI example (Figure 4)
bash src/runners/wadi_example.sh
```

---

## Mapping Paper Outcomes to Artifact Outputs

### Overview

This section provides a clear mapping between each result in the paper and the corresponding artifact output. Each functional outcome is labeled (F1, F2, ..., Fn) for easy reference. All results can be generated using the following command, which contains all of the evaluation scripts for each outcome:

```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/figures:/app/figures \
    my-evaluation:latest
```

To see the specific scripts being executed, we refer the user to observe `run_evaluation.sh`. Below, we'll match each script (and its output) to its functional outcome in the paper.


---

### F1: Classification Performance Metrics

**Mapping to Paper:** Our model's classification results in Table 4

**Description:** AUC and F1 score metrics across all three datasets (PMU, SWAT, WADI). Additionally, Precision and Recall can be found in Table 7 in the Appendix. Each metric also has its standard deviations given.

**Script:** `src/get_results.py`

**Expected Output Location:** `results/results.txt`

**These results will be shown near the top of the file under "WENFlow Detection and Efficiency Results (Tables 4, 5)". Results are given as averages across five random seeds.**

---

### F2: Efficiency Metrics

**Mapping to Paper:** Our model's efficiency results in Table 5

**Description:** Various efficiency metrics including GPU memory, inference time, and parameter counts. 

**Script:** `src/get_results.py`

**Expected Output Location:** `results/results.txt`

**Variability:** There will be some variability for metrics such as inference time due to user-specific hardware differences.

**These results will be shown under "WENFlow Detection and Efficiency Results (Tables 4, 5)" near the top of the file.**

---

### F3: PMU Histogram Example

**Mapping to Paper:** Figure 3

**Description:** Visualization of anomaly detection on PMU dataset, showing likelihood distributions for normal vs. anomalous events.

**Script:** `src/runners/likelihoods_pmu.sh`

**Expected Output Location:** `figures/pmu_histogram.png`

**Variability:** Some minor variability in the plotting since we use python's matplotlib library for plotting in this evaluation. In the paper, we instead opt for LaTeX's PGFPlots, which enhances visual quality.

---

### F4: WADI Example

**Mapping to Paper:** Figure 4

**Description:** A figure showing 4 separate plots of data taken from the WADI dataset. 

**Scripts:** `src/runners/wadi_example.sh` for generating plotting data and `src/visualize_wadi.py` for visualization.

**Expected Output Location:** `figures/wadi_example.png`


**Variability:** Some minor variability in the plotting since we use python's matplotlib library for plotting in this evaluation. In the paper, we instead opt for LaTeX's PGFPlots, which enhances visual quality.


---

### F5: Scaling Analysis 


**Mapping to Paper:** Figure 5

**Description:** A figure showing memory consumption after changing input dimension size. Four separate plots for different WENFlow models should be present in the figure. 

**Scripts:** `src/runners/test_scaling.sh` for generating the data for plotting and `src/visualize_scaling.py` for visualization.

**Expected Output Location:** `figures/memory_scaling.png`

**Variability:** Some minor variability in the plotting since we use python's matplotlib library for plotting in this evaluation. In the paper, we instead opt for LaTeX's PGFPlots, which enhances visual quality.


---

### F6: Ablation Study Results


**Mapping to Paper:** Figure 6

**Description:** AUC, inference time, and parameter count results for 3 different ablation models (WF\A, WF\W, and RealNVP). 

**Script:** `src/get_results.py`

**Expected Output Location:** `results/results.txt`

**Variability:** Some minor variability for inference times.

**These results will be shown under "WENFlow Ablation Results (Table 6)" in the results file.**
---

### F7: Sensitivity Analyses


**Mapping to Paper:** Figures 6, 7, and 8

**Description:** Three figures showing AUC values for our model using varying k values (figure 6), wavelet types (figure 7), and window sizes (figure 8)

**Script:** `src/get_results.py`

**Expected Output Location:** `figures/sensitivity_k.png`, `figures/sensitivity_wavelet.png`, and `figures/sensitivity_window.png`.

**Variability:** Some minor variability in the plotting since we use python's matplotlib library for plotting in this evaluation. In the paper, we instead opt for LaTeX's PGFPlots, which enhances visual quality.


---

### F8: Cascade Analysis


**Mapping to Paper:** Figure 9

**Description:** Three figures: one showing mean top-k overlap ratios across consecutive windows, one showing top-k overlaps across different cascade stages, and one showing the distribution of top-k overlap values for synthetic events in the simulation.

**Script:** `src/cascade_example.py`

**Expected Output Location:** `figures/cascade_progression.png` (figure 9 (a) and (b)) and `figures/topk_overlap_histogram.png` (figure 9 (c))

**Variability:** Some minor variability in the plotting since we use python's matplotlib library for plotting in this evaluation. In the paper, we instead opt for LaTeX's PGFPlots, which enhances visual quality.

---

## License

See LICENSE file in the repository for details.

---


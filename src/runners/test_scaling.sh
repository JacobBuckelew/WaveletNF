k_values="0.01 0.05 0.1 0.25"
for k in $k_values; do
    CUDA_VISIBLE_DEVICES=0 python3 src/test_scaling.py\
        --batch_size=64\
        --window_size=64\
        --num_blocks=1\
        --k=$k\
        --st_units=32\
        --gpu\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=64\
        --heads=1\
        --seed=6\
        --name=WaveletNF_PMU_seed_6
    done
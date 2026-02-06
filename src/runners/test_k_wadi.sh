k_values="0.05 0.10 0.15 0.20 0.25"
for seed in {6..8}; do
    for k_value in $k_values; do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=512\
        --window_size=32\
        --gpu\
        --k=${k_value}\
        --num_blocks=1\
        --st_units=16\
        --dataset=WADI\
        --wavelet_type=coif2\
        --N=32\
        --heads=1\
        --seed=${seed}\
        --name=k${k_value}_wadi_seed_${seed}
    done
done
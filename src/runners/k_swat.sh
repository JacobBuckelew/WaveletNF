k_values="0.10 0.15 0.20 0.25"
for seed in {6..8}; do
    for k_value in $k_values; do
        CUDA_VISIBLE_DEVICES=3 python3 ../train.py\
        --batch_size=512\
        --window_size=64\
        --lr=0.001\
        --k=${k_value}\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --epochs=10\
        --dataset=SWAT\
        --wavelet_type=coif1\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=k${k_value}_swat_seed_${seed}
    done
done
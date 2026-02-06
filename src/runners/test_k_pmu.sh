k_values="0.05 0.10 0.15 0.20 0.25"
for seed in {6..8}; do
    for k_value in $k_values; do
        CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
            --batch_size=512\
            --window_size=16\
            --num_blocks=1\
            --k=${k_value}\
            --st_units=32\
            --gpu\
            --dataset=PMU\
            --wavelet_type=db2\
            --N=16\
            --heads=1\
            --seed=${seed}\
            --name=k${k_value}_PMU_seed_${seed}
    done
done

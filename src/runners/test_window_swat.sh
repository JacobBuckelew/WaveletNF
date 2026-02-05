sizes="16 32 48 96"
for seed in {6..8}; do
    for size in $sizes; do
        CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=512\
        --window_size=${size}\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --dataset=SWAT\
        --wavelet_type=coif1\
        --N=${size}\
        --heads=1\
        --seed=${seed}\
        --name=window${size}_swat_seed_${seed}
    done
done
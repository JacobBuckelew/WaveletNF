sizes="16 32 48 64 96"
for seed in {6..8}; do
    for size in $sizes; do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
       --batch_size=512\
        --window_size=${size}\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=${size}\
        --heads=1\
        --seed=${seed}\
        --name=window${size}_PMU_seed_${seed}
    done
done
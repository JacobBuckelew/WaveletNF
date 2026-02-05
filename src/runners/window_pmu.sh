sizes="32 48 64 96"
for seed in {6..8}; do
    for size in $sizes; do
    CUDA_VISIBLE_DEVICES=2 python3 ../train.py\
       --batch_size=512\
        --window_size=${size}\
        --lr=0.006\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --epochs=50\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=${size}\
        --heads=1\
        --seed=${seed}\
        --name=window${size}_PMU_seed_${seed}
    done
done
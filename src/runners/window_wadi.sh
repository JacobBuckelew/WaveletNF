sizes="16 48 64 96"
for seed in {6..8}; do
    for size in $sizes; do
    CUDA_VISIBLE_DEVICES=2 python3 ../train.py\
        --batch_size=512\
        --window_size=${size}\
        --lr=0.001\
        --gpu\
        --k=0.10\
        --num_blocks=1\
        --st_units=16\
        --epochs=10\
        --dataset=WADI\
        --wavelet_type=coif2\
        --N=${size}\
        --heads=1\
        --seed=${seed}\
        --name=window${size}_wadi_seed_${seed}
    done
done
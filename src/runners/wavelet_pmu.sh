wavelets="haar db1 coif2 coif1"
for seed in {6..8}; do
    for wavelet in $wavelets; do
        CUDA_VISIBLE_DEVICES=3 python3 ../train.py\
        --batch_size=512\
        --window_size=16\
        --lr=0.006\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --epochs=50\
        --dataset=PMU\
        --wavelet_type=${wavelet}\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=${wavelet}_PMU_seed_${seed}
    done
done
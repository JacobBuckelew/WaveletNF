wavelets="haar db1 db2 coif2 coif1"
for seed in {6..8}; do
    for wavelet in $wavelets; do
        CUDA_VISIBLE_DEVICES=0 python3 src/test.py\
        --batch_size=512\
        --window_size=16\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --dataset=PMU\
        --wavelet_type=${wavelet}\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=${wavelet}_PMU_seed_${seed}
    done
done
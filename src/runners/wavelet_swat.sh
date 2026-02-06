wavelets="haar db1 db2 coif1 coif2"
for seed in {6..8}; do
    for wavelet in $wavelets; do
        CUDA_VISIBLE_DEVICES=3 python3 ../train.py\
        --batch_size=512\
        --window_size=64\
        --lr=0.001\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --epochs=10\
        --dataset=SWAT\
        --wavelet_type=${wavelet}\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=${wavelet}_swat_seed_${seed}
    done
done
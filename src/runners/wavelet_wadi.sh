wavelets="haar db1 db2 coif1 coif2"
for seed in {6..8}; do
    for wavelet in $wavelets;do
        CUDA_VISIBLE_DEVICES=2 python3 ../train.py\
        --batch_size=512\
        --window_size=32\
        --lr=0.001\
        --gpu\
        --k=0.10\
        --num_blocks=1\
        --st_units=16\
        --epochs=10\
        --dataset=WADI\
        --wavelet_type=${wavelet}\
        --N=32\
        --heads=1\
        --seed=${seed}\
        --name=${wavelet}_wadi_seed_${seed}
    done
done
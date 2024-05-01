for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.001\
        --num_blocks=1\
        --st_units=32\
        --epochs=25\
        --dataset=SWAT\
        --wavelet_type=haar\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_swat_seed_${seed}
done
for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=2 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.001\
        --num_blocks=2\
        --lam=0.99\
        --st_units=64\
        --epochs=20\
        --dataset=WADI\
        --wavelet_type=haar\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_wadi_seed_${seed}
done
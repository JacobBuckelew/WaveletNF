for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=1\
        --window_size=64\
        --num_blocks=2\
        --lam=0.8\
        --stride_size=10\
        --st_units=64\
        --dataset=WADI\
        --wavelet_type=haar\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_wadi_seed_${seed}
done
for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 src/train.py\
        --batch_size=512\
        --window_size=64\
        --lr=0.001\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --epochs=10\
        --dataset=SWAT\
        --wavelet_type=coif1\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_swat_seed_${seed}
done
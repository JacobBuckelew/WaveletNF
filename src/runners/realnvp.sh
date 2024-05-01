for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.001\
        --num_blocks=2\
        --st_units=32\
        --epochs=25\
        --dataset=PSM\
        --wavelet=0\
        --attention=0\
        --heads=1\
        --seed=${seed}\
        --name=RealNVP_psm_seed_${seed}
done
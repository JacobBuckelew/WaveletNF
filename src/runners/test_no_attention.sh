for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=256\
        --window_size=64\
        --num_blocks=2\
        --st_units=32\
        --dataset=PSM\
        --wavelet_type=haar\
        --attention=0\
        --N=64\
        --seed=${seed}\
        --name=WENFA_psm_seed_${seed}
done
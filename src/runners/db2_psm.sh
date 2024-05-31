for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.001\
        --num_blocks=2\
        --st_units=32\
        --epochs=20\
        --lam=0.50\
        --dataset=PSM\
        --wavelet_type=db2\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=db2_psm_seed_${seed}
done

for seed in {6..10}
do
        CUDA_VISIBLE_DEVICES=1 python3 ../test.py\
                --batch_size=256\
                --window_size=64\
                --num_blocks=2\
                --st_units=32\
                --lam=0.5\
                --dataset=PSM\
                --wavelet_type=db2\
                --N=64\
                --heads=1\
                --seed=${seed}\
                --name=db2_psm_seed_${seed}
done
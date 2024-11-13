for seed in {6..8}
do  
    CUDA_VISIBLE_DEVICES=0 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.006\
        --num_blocks=1\
        --st_units=32\
        --epochs=40\
        --lam=0.1\
        --dataset=PMU\
        --wavelet_type=haar\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=lambda0.1_PMU_seed_${seed}
done

for seed in {6..8}
do  
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=256\
        --window_size=64\
        --num_blocks=1\
        --st_units=32\
        --lam=0.1\
        --dataset=PMU\
        --wavelet_type=haar\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=lambda0.1_PMU_seed_${seed}
done
for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=3 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.006\
        --num_blocks=1\
        --st_units=32\
        --epochs=40\
        --lam=0.75\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=db2_PMU_seed_${seed}
done


for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=3 python3 ../test.py\
        --batch_size=256\
        --window_size=64\
        --num_blocks=1\
        --st_units=32\
        --lam=0.75\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=db2_PMU_seed_${seed}
done
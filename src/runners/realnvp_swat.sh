for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../train.py\
        --batch_size=256\
        --window_size=64\
        --lr=0.0015\
        --num_blocks=1\
        --st_units=64\
        --lam=0.8\
        --epochs=15\
        --dataset=SWAT\
        --N=64\
        --attention=0\
        --wavelet=0\
        --seed=${seed}\
        --name=RealNVP_swat_seed_${seed}
done

for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../test.py\
        --batch_size=256\
        --window_size=64\
        --num_blocks=1\
        --st_units=64\
        --dataset=SWAT\
        --lam=0.8\
        --attention=0\
        --wavelet=0\
        --N=64\
        --seed=${seed}\
        --name=RealNVP_swat_seed_${seed}
done
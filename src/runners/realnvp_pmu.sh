for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../train.py\
        --batch_size=512\
        --window_size=16\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --wavelet=0\
        --epochs=1\
        --lr=0.006\
        --attention=0\
        --gpu\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=RealNVP_PMU_seed_${seed}
done

for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../test.py\
        --batch_size=512\
        --window_size=16\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --wavelet=0\
        --attention=0\
        --gpu\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=RealNVP_PMU_seed_${seed}
done
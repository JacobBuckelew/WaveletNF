for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 src/test.py\
        --batch_size=512\
        --window_size=16\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --wavelet=0\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=WENFW_PMU_seed_${seed}
done
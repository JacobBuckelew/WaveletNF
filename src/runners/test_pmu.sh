for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 src/test.py\
        --batch_size=256\
        --window_size=16\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_PMU_seed_${seed}
done
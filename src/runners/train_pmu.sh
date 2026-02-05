for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../train.py\
        --batch_size=512\
        --window_size=16\
        --lr=0.006\
        --num_blocks=1\
        --k=0.10\
        --st_units=32\
        --gpu\
        --epochs=50\
        --dataset=PMU\
        --wavelet_type=db2\
        --N=16\
        --heads=1\
        --seed=${seed}\
        --name=WaveletNF_PMU_seed_${seed}
done
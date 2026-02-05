for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=512\
        --window_size=32\
        --gpu\
        --k=0.10\
        --num_blocks=1\
        --st_units=16\
        --dataset=WADI\
        --wavelet=1\
        --attention=0\
        --wavelet_type=coif2\
        --N=32\
        --heads=1\
        --seed=${seed}\
        --name=WENFA_wadi_seed_${seed}
done
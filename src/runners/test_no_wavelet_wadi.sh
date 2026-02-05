for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=512\
        --window_size=32\
        --gpu\
        --k=0.10\
        --num_blocks=1\
        --st_units=16\
        --wavelet=0\
        --attention=1\
        --dataset=WADI\
        --wavelet_type=coif2\
        --N=32\
        --heads=1\
        --seed=${seed}\
        --name=WENFW_wadi_seed_${seed}
done
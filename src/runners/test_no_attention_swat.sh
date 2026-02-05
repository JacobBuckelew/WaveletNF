for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
        --batch_size=512\
        --window_size=64\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --dataset=SWAT\
        --wavelet=1\
        --attention=0\
        --wavelet_type=coif1\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WENFA_swat_seed_${seed}
done

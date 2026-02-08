for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python3 src/test.py\
        --batch_size=512\
        --window_size=64\
        --num_blocks=1\
        --gpu\
        --k=0.05\
        --st_units=16\
        --dataset=SWAT\
        --N=64\
        --attention=0\
        --wavelet=0\
        --seed=${seed}\
        --name=RealNVP_swat_seed_${seed}
done
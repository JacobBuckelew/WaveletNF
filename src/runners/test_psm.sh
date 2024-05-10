for seed in {6..10}
do
        CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
                --batch_size=256\
                --window_size=64\
                --num_blocks=2\
                --st_units=32\
                --dataset=PSM\
                --wavelet_type=haar\
                --N=64\
                --heads=1\
                --seed=${seed}\
                --name=WaveletNF_psm_seed_${seed}
done
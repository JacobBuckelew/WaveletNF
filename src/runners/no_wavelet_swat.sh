for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../train.py\
        --batch_size=512\
        --window_size=64\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --dataset=SWAT\
        --wavelet_type=coif1\
        --wavelet=0\
        --epochs=3\
        --lr=0.001\
        --attention=1\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WENFW_swat_seed_${seed}
done
for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python3 ../test.py\
        --batch_size=512\
        --window_size=64\
        --k=0.05\
        --gpu\
        --num_blocks=1\
        --st_units=16\
        --dataset=SWAT\
        --wavelet_type=coif1\
        --wavelet=0\
        --attention=1\
        --N=64\
        --heads=1\
        --seed=${seed}\
        --name=WENFW_swat_seed_${seed}
done
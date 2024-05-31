for seed in {6..10}
do  
    for i in $(seq 0.2 0.1 0.9)
    do
        CUDA_VISIBLE_DEVICES=0 python3 ../train.py\
            --batch_size=256\
            --window_size=64\
            --lr=0.001\
            --num_blocks=2\
            --st_units=32\
            --epochs=20\
            --lam=${i}\
            --dataset=PSM\
            --wavelet_type=haar\
            --N=64\
            --heads=1\
            --seed=${seed}\
            --name=lambda${i}_psm_seed_${seed}
    done
done

for seed in {6..10}
do  
    for i in $(seq 0.2 0.1 0.9)
    do
        CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
            --batch_size=256\
            --window_size=64\
            --num_blocks=2\
            --st_units=32\
            --lam=${i}\
            --dataset=PSM\
            --wavelet_type=haar\
            --N=64\
            --heads=1\
            --seed=${seed}\
            --name=lambda${i}_psm_seed_${seed}
    done
done
for seed in {6..10}
do
        CUDA_VISIBLE_DEVICES=0 python3 ../test.py\
                --n_blocks=1\
                --batch_size=256\
                --dataset=SWAT\
                --window_size=60\
                --train_split=0.8\
                --seed=${seed}\
                --name=MTGFLOW_swat_seed_${seed}
done
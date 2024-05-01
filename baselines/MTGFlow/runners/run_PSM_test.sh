for seed in {6..10}
do
        CUDA_VISIBLE_DEVICES=1 python3 ../test.py\
                --n_blocks=2\
                --batch_size=256\
                --window_size=60\
                --train_split=0.8\
                --dataset=PSM\
                --seed=${seed}\
                --name=MTGFLOW_psm_seed_${seed}
done

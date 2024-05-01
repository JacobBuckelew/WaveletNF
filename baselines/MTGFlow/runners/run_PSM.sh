for seed in {6..10}
do
        CUDA_VISIBLE_DEVICES=3 python3 ../main.py\
                --n_blocks=2\
                --batch_size=256\
                --window_size=60\
                --name=MTGFLOW_psm_seed_${seed}\
                --train_split=0.8\
                --dataset=PSM\
                --seed=${seed}
done
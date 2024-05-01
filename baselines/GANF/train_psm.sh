for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python -u train_model.py\
        --dataset=psm\
        --name=GANF_psm_seed_${seed}\
        --train_epochs=40\
        --lr=0.002\
        --seed=${seed}\
        --hidden_size=16\
        --batch_size=256\
        --n_epochs=1\
        --n_blocks=2
done

for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python -u train_model.py\
        --dataset=swat\
        --name=GANF_swat_seed_${seed}\
        --train_epochs=40\
        --hidden_size=32\
        --seed=${seed}\
        --n_epochs=1\
        --n_blocks=1\
        --lr=0.002\
        --batch_size=512\
        --weight_decay=0.0005
done



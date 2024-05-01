for seed in {6..10}
do 
    CUDA_VISIBLE_DEVICES=0 python -u test_model.py\
        --dataset=swat\
        --seed=${seed}\
        --batch_size=256\
        --name=GANF_swat_seed_${seed}\
        --hidden_size=32\
        --n_blocks=1
done

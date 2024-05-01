for seed in {6..10}
do
    CUDA_VISIBLE_DEVICES=0 python -u test_model.py\
        --dataset=psm\
        --seed=${seed}\
        --name=GANF_psm_seed_${seed}\
        --n_blocks=2\
        --hidden_size=16
done
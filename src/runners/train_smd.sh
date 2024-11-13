for file in /home/jbuckelew/workspace/WaveletNF/data/smd/*machine*
do 
    var=${file##*/}
    # echo $var
    echo ${var%%_*}
    for seed in {6..10}
    do
        CUDA_VISIBLE_DEVICES=1 python3 -u ../train.py\
            --num_blocks=2\
            --batch_size=256\
            --window_size=64\
            --st_units=64\
            --lam=0.80\
            --wavelet_type=haar\
            --N=64\
            --epochs=25\
            --lr=0.002\
            --dataset=${var%%_*}\
            --seed=${seed}\
            --name=WaveletNF_smd_${var%%_*}_seed_${seed}
    done
done
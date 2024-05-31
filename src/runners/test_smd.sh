for file in /home/jbuckelew/workspace/WaveletNF/data/smd/*train*
do 
    var=${file##*/}
    # echo $var
    echo ${var%_*}
    for seed in {6..10}
    do
        CUDA_VISIBLE_DEVICES=1 python3 -u ../test.py\
            --num_blocks=2\
            --batch_size=256\
            --window_size=64\
            --stride_size=64\
            --st_units=64\
            --lam=0.80\
            --wavelet_type=haar\
            --N=64\
            --dataset=${var%_*}\
            --seed=${seed}\
            --name=WaveletNF_smd_${var%_*}_seed_${seed}
    done
done
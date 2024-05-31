for file in ../../data/smd/*train*
do
    var=${file##*/}
    CUDA_VISIBLE_DEVICES=3 python3 ../hyperparameter_tuning.py\
        --dataset=${var%_*}\
        --wavelet_type=haar\
        --window_size=64\
        --N=64\
        --heads=1\
        --seed=7
done
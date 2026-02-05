sizes="96 32 48 64 16"
k_values="0.05 0.10 0.15 0.20 0.25"
wavelets="haar db1 db2 coif1 coif2"
for seed in {6..8}; do
        for size in $sizes; do
                for k_value in $k_values; do
                        for wavelet in $wavelets; do
                                CUDA_VISIBLE_DEVICES=1 python3 ../train.py\
                                        --batch_size=512\
                                        --window_size=${size}\
                                        --lr=0.006\
                                        --num_blocks=1\
                                        --k=${k_value}\
                                        --st_units=32\
                                        --gpu\
                                        --epochs=50\
                                        --dataset=PMU\
                                        --wavelet_type=${wavelet}\
                                        --N=${size}\
                                        --heads=1\
                                        --seed=${seed}\
                                        --name=window${size}_wavelet${wavelet}_k${k_value}_PMU_seed_${seed}
                        done
                done
        done
done
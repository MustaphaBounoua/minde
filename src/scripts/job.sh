#!/bin/bash
source myvenv/bin/activate

export CUDA_VISIBLE_DEVICES="0" 
python -m scripts.run_minde --importance_sampling --use_ema --test_epoch 10000 --task_0 0 --task_n 4 &



for i in {0..20}
do
    export CUDA_VISIBLE_DEVICES="$k"  
    k=$((i%4))
    m=$((i+1))
    echo "GPU $k:"
    python -m src.scripts.run_minde  --importance_sampling --use_ema --test_epoch 10000 --task_0 $i --task_n $m --results_dir "results_c" &
    pids[${i}]=$!
done
for pid in ${pids[*]}; do
    wait $pid
done

for i in {20..39}
do
    export CUDA_VISIBLE_DEVICES="$k"  
    k=$((i%4))
    m=$((i+1))
    echo "GPU $k:"
    python -m src.scripts.run_minde  --importance_sampling --use_ema --test_epoch 10000 --task_0 $i --task_n $m --results_dir "results_c" &
    pids[${i}]=$!
done
for pid in ${pids[*]}; do
    wait $pid
done
echo "hello"




for i in {0..20}
do
    export CUDA_VISIBLE_DEVICES="$k"  
    k=$((i%4))
    m=$((i+1))
    echo "GPU $k:"
    python -m src.scripts.run_minde --type "j"  --importance_sampling --use_ema --test_epoch 10000 --task_0 $i --task_n $m --results_dir "results_j" &
    pids[${i}]=$!
done
for pid in ${pids[*]}; do
    wait $pid
done

for i in {20..39}
do
    export CUDA_VISIBLE_DEVICES="$k"  
    k=$((i%4))
    m=$((i+1))
    echo "GPU $k:"
    python -m src.scripts.run_minde --type "j" --importance_sampling --use_ema --test_epoch 10000 --task_0 $i --task_n $m --results_dir "results_j" &
    pids[${i}]=$!
done
for pid in ${pids[*]}; do
    wait $pid
done
echo "hello"
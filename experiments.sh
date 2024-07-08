#!/bin/bash

for i in $(seq 1 5)
do
   echo "Ejecutando la iteración $i..."
    python3 train.py --exp_name "reconstructCUBE_$i" --model "fcn" --datadir "LIB-HSI" \
    --gpu 1 --epochs 150 --batch_size 16 --lr 3e-3  --usergb --reconstruct --regularize \
    --wandb --group "reconstructCUBE"
   echo "Iteración $i completada."
done

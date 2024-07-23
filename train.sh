python3 train.py --exp_name "rgb_FCNpretrained" --model "unetsmp" --datadir "LIB-HSI" \
   --gpu 0 --epochs 150 --batch_size 16 --lr 3e-3  --usergb  \
   --wandb
python3 train.py --exp_name "fcnrepeatrgb" --model "fcn" --datadir "LIB-HSI" \
    --gpu 0 --epochs 150 --batch_size 16 --lr 3e-3 --wandb  --usergb --group "rgb"
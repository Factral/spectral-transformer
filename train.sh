python3 train.py --exp_name "unet" --model "unet" --datadir "LIB-HSI" \
    --gpu 0 --epochs 150 --batch_size 8 --lr 3e-3 --wandb  --usergb --group "rgb"
CUDA_VISIBLE_DEVICES=1 python3 tools/train.py \
     local_configs/segformer_mit-prompt-b3_200epochs_hsi-512x512.py \
     segformer --amp --wandb
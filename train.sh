CUDA_VISIBLE_DEVICES=1 python3 tools/train.py \
     local_configs/segformer_mit-b0_8xb2-160k_hsi-512x512.py \
     segformer --amp --wandb
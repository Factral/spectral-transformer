CUDA_VISIBLE_DEVICES=1 python tools/train.py \
     local_configs/segformer_mit-b0_8xb2-160k_hsi-512x512.py \
     segformer --amp
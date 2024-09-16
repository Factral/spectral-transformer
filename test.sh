CUDA_VISIBLE_DEVICES=0 python tools/test.py \
     local_configs/segformer_mit-b0_8xb2-160k_hsi-512x512.py \
     output/segformer/exp_20240915-213824/epoch_200.pth \
     --show-dir ./results

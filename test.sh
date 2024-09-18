CUDA_VISIBLE_DEVICES=0 python tools/test.py \
     local_configs/segformer_mit-prompt-spectral-b3_200epochs_hsi-512x512.py \
     output/segformer/exp_20240916-220506/epoch_200.pth \
     --show-dir ./results

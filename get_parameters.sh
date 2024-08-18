CUDA_VISIBLE_DEVICES=0 python tools/analysis_tools/get_flops.py \
     local_configs/segformer_mit-b0_8xb2-160k_hsi-512x512.py \
     --shape 512 512

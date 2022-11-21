# pytorch==1.4  
# python==3.6

1. Start training

CUDA_VISIBLE_DEVICES=0 python train.py -config run_config/train-cross-example.json

1. Start decoding
CUDA_VISIBLE_DEVICES=0 python translate.py -config run_config/decode-cross-example.json

pytorch>=1.0, python==3.6

1. Start training

CUDA_VISIBLE_DEVICES=0 python train.py -config run_config/train-idx-cross-exmaple.josn

2. Start decoding

CUDA_VISIBLE_DEVICES=0 python translate.py -config run_config/decode-cross-example.json

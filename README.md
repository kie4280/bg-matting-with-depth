# Extend the work of BackgroundMattingV2
The original work is [here](https://github.com/PeterL1n/BackgroundMattingV2)  
We use [Intel MiDaS](https://github.com/intel-isl/MiDaS) as ground truth for training our own depth estimator

## Changes made
* Add a new depth channel to the end of the mattingbase decoder
* Add random pedestrians to the video input so that the model can learn to use depth as a feature, not only relying on the background cues for matting.

## Inference
```
python inference_custom.py \
        --V2-model-checkpoint \
            /home/kie/research/pretrained/V2-model.pth \
        --Midas-model-checkpoint \
            /home/kie/research/pretrained/intel-MiDas-model.pt \
        --Midas-model-type large
```

## Training
```
CUDA_VISIBLE_DEVICES=1 python V2/train_base.py \
        --dataset-name photomatte85 \
        --model-backbone resnet50 \
        --model-name mattingbase-resnet50-photomatte85 \
        --model-pretrain-initialization "/home/kie/research/pretrained/best_deeplabv3_resnet50_voc_os16.pth" \
        --epoch-end 8
```



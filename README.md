# Extend the work of BackgroundMattingV2
The original work is [here](https://github.com/PeterL1n/BackgroundMattingV2)  
We use [Intel MiDaS](https://github.com/intel-isl/MiDaS) for our depth estimator

## Changes made
* Add depth filter to the front and the end of the original model

## Inference
```
python inference_custom.py \
        --V2-model-checkpoint \
            /home/kie/research/pretrained/V2-model.pth \
        --Midas-model-checkpoint \
            /home/kie/research/pretrained/intel-MiDas-model.pt \
        --Midas-model-type large
```

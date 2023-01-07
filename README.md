# PyTorch losses and metrics
- Write losses and metrics for Detection and Segmentation task

### Segmentation task  
- Weighted Cross Entropy Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/weighted_celoss.py)
- Ohem Cross Entropy Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/ohem.py)
- Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/focal_loss_seg.py)
- Dice Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/dice.py)
- Mean IOU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/miou.py)
- Pixel Accuracy [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/segmentation/pixel_accuracy.py)

### Detection task
- IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/iou.py)
- Generalized IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/giou.py) [(paper)](https://arxiv.org/abs/1911.08287)
- Distance IoU [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/diou.py) [(paper)](https://arxiv.org/abs/1911.08287)
- Focal Loss [(code)](https://github.com/Sangh0/pytorch-loss-metric/blob/main/detection/focal_loss.py) [(paper)](https://arxiv.org/abs/1708.02002)
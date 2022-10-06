# Object-Detection

## Model Inference 

please follow the [Instruction](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) to install mmdetection.

Download the [checkpoints](https://drive.google.com/file/d/1tyysydkQy0L5IWoagDfKLB-w74GX6D9G/view?usp=sharing)
Run
````Python
python inference.py
````



## Model Training 

Download the our waste images [dataset](https://drive.google.com/drive/folders/1-23NmZPe4av56-1u2A8Ev4_4eBlmyvc-?usp=sharing) (with [training annotations](https://drive.google.com/file/d/1WNo_BLJYWYewXbUVuEJjVyR2hPHOYuXq/view?usp=sharing) and [testing annotations](https://drive.google.com/file/d/1WS9f4ZxF9XT6KUM0Uoj7qH3qayCjZdIl/view?usp=sharing
))

then, follow the [Instruction](https://mmdetection.readthedocs.io/en/stable/get_started.html#installation) to install mmdetection. Follow the instruction for model training with custom dataset



## Performance
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.733
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1000 ] = 0.899
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1000 ] = 0.764
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.650
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1000 ] = 0.782
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.667
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.805
```

## Detection Visualization
for the full visualization results, please click this [link](https://drive.google.com/drive/folders/1GF0rOI4xoRuV1Co9jTWBg1gP6LlzpvfF?usp=sharing).

![Alt text](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/drink_cup_pepsi%20(2).JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/drink_cup_pepsi%20(80).JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/kettle_corn_bag%20(15).JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/napkins%20(48).JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/nestle-51.JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/nestle-71.JPG)
![ ](https://github.com/zotbins/Zotbins-Waste-Recognition/blob/main/Object%20Detection/results/paper_box-9.jpg)

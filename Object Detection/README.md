# Object-Detection

Prerequisites:
For model inference, please follow the [Instruction][] to install mmdetection.
[Instruction]:https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md

Download the [checkpoints][]
[checkpoints]:https://drive.google.com/file/d/1tyysydkQy0L5IWoagDfKLB-w74GX6D9G/view?usp=sharing

````Python
python inference.py
````

For model training, 
Download the our waste images [dataset][] (with [training annotations][] and [testing annotations][])
[dataset][]:https://drive.google.com/drive/folders/1-23NmZPe4av56-1u2A8Ev4_4eBlmyvc-?usp=sharing
[training annotations][]:https://drive.google.com/file/d/1WNo_BLJYWYewXbUVuEJjVyR2hPHOYuXq/view?usp=sharing
[testing annotations][]:https://drive.google.com/file/d/1WS9f4ZxF9XT6KUM0Uoj7qH3qayCjZdIl/view?usp=sharing

then, follow the [Instruction][] to install mmdetection. Follow the instruction for model training with custom dataset
[Instruction]:https://mmdetection.readthedocs.io/en/stable/get_started.html#installation



Performance
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
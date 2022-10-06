from mmdet.apis import init_detector, inference_detector

config_file = 'configs/gfl_r50_fpn_1x_waste/gfl_r50_fpn_1x_waste.py'
checkpoint_file = 'checkpoints/gfl_r50_fpn_1x_waste/latest.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
inference_detector(model, 'demo/coffee_cup_no_logo (1).JPG')
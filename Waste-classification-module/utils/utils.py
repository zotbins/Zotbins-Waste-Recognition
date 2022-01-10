import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]) 



def predict(img, pretrained='no_aug.pth'):
    """
    Returns the predictions of the images
    
    :param img: RGB images of size
    :param pred_num: the number of prediction to return (default 5)
    :param model: model for prediction (default resnet18)
    :param pretrained: pretrained model (default resnet18_pretrained)
    
    :return: a list of predictions (2d list)
    	for each prediction,
    	[
    	0: index of the image,
    	1: index of the predictions
    	2: confidence of each predictions
    	]
    """

    reshaped = preprocess(img)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnext50_32x4d(pretrained=False, num_classes=3)
    model.load_state_dict(torch.load("pretrained/"+pretrained, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    
    outputs = model(reshaped.unsqueeze(0).to(device))
    return class_map[torch.argmax(outputs, 1).cpu().item()]
   
 
class_map = {0: "Landfill", 1: "Recycle", 2: "Compost"}

'''
class_map = {0: 'Nekter Plastic Cup (dirty)',
         1: 'plasic wrapper',
         2: 'Yogurt Yoplait Cup (dirty)',
         3: 'Straws',
         4: 'Soylent Bottle (Chai)',
         5: 'Paper Boat (dirty)',
         6: 'Jamba Juice Plastic Cup',
         7: 'Black Plastic Fork',
         8: 'Taco Patio Water Cup',
         9: 'Chip Bags',
         10: 'Wrappers',
         11: 'Coca-cola Can',
         12: 'Starbucks Grande Iced Cold Cup',
         13: 'Brown Napkin(Crumpled)',
         14: 'Phoenix Grill Clear Drink Cup',
         15: 'Straw Wrap Paper',
         16: 'Napkins',
         17: 'Panda Black Bowl',
         18: 'Starbucks Green Straw',
         19: 'subway paper food wrapper',
         20: 'paper food container',
         21: 'Spoiling Banana Peel',
         22: 'Plastic Water Cup',
         23: 'Starbucks Pastry Wrapper',
         24: "Wendy's Soda Lid",
         25: 'White bags',
         26: 'Einstein Bro’s Wrapper',
         27: 'Starbucks Iced Grande Cup (dirty)',
         28: 'Paper containers',
         29: 'plasic food container',
         30: 'plasic cup',
         31: 'Subway bag (crumpled)',
         32: 'Plastic cup with fruit (empty)',
         33: 'Gatorade bottle',
         34: 'paper cup with plasic or paper lid',
         35: 'Compostable Bowl (dirty)',
         36: 'paper cup',
         37: 'Boxes',
         38: 'plasic cup with plasic or paper lid',
         39: 'Ecogrounds Hot Coffee Cup',
         40: 'Soup container (dirty)',
         41: 'pepsi cup',
         42: 'Starbucks Hot Venti Cup',
         43: 'White Paper Plate',
         44: 'Crumpled Saran Wrap ',
         45: 'Panda Chopsticks Wrapper',
         46: 'Brown Napkin(Unfolded new)',
         47: 'Panda Water Cup',
         48: 'Aluminum Foil',
         49: 'Subway Sandwich Wrapper',
         50: 'Black Plastic Spoon',
         51: 'Chick Fil A Styrofoam Cup',
         52: 'Phoenix Grill Checkered Liner',
         53: 'Pepsi Photos',
         54: 'Brown bags',
         55: 'Subway Bag, Clam, Pepsi Cup',
         56: 'Jamba Juice Plastic Cup (dirty)',
         57: 'Subway plastic drink cup',
         58: 'Compostable Rectangle To-Go-Box',
         59: 'Compostable plate (clean)',
         60: 'Plastic containers',
         61: 'Plastic side cup (dirty)',
         62: 'Starbucks Venti Frap Cup',
         63: 'Starbucks Can',
         64: 'Black Plastic Knife',
         65: 'Starbucks Sippy Cup',
         66: 'Compostable Square To-Go Box',
         67: 'Subway Cookie Bag',
         68: 'Starbucks Green Straw wrapper',
         69: 'plasic or paper lid',
         70: 'Clear Plastic Bottle (empty)',
         71: 'Yogurt Cup (dirty)',
         72: 'Pepsi Soda Paper Cup',
         73: 'paper straw',
         74: 'Plastic Salad Bowl (Dirty)',
         75: 'Condiment Packets',
         76: 'Water Bottle (Camera)'}
'''
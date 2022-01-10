from test.utils import loadTestData
from utils.utils import predict
import matplotlib.pyplot as plt
import random


'''
Note: 
The prediction should take shorter than a second without GPU
If the program takes a very long time to run, 
most likely it spends much time in configuration
'''

print("loading data...")
imgs = loadTestData(size=(256,256))
img = random.choice(imgs)


#make prediction
#the img parameter should be a PIL image object with size (256,256,3)
prediction = predict(img)

#visualize the result
plt.imshow(img)
plt.title("Prediction: " + prediction)
plt.show()



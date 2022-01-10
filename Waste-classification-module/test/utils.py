import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

def loadTestData(size=(64,64)):
  test_data = []
  fdir = "test"
  for filename in os.listdir(fdir):
    name, extension = os.path.splitext(filename)
    if extension.lower() == '.jpg':
      numpy_file = plt.imread(fdir + "/" + filename)
      img = Image.fromarray(numpy_file, "RGB")
      reshaped_img = img.resize(size)
      test_data.append(reshaped_img)
  return test_data

import os

import cv2
import keras
import numpy
import variables 
import matoplotlib.pyplot

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imread, imresize, imshow

from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

img_rows, img_cols = vars.img_rows, vars.img_cols
batch_size = vars.batch_size
num_classes = vars.num_classes
epochs = vars.epochs
model_json_path = vars.model_json_path
model_path = vars.model_path
prediction_file_dir_path = vars.prediction_file_dir_path

path = './FEATURE-BASED-IMAGES/'

data = []
labels = []

for folder, subfolders, files in os.walk(path):
    for name in files:
        if name.endswith('.jpg'):
            x = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
            x = imresize(x, (img_rows, img_cols))
            __, x = cv2.threshold(x, 220, 255, cv2.THRESH_BINARY)

            # dilate
            morph_size = (2, 2)
            pass
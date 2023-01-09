from __future__ import print_function

import os
import cv2
import keras
import numpy
import variables as vars
import pickle
import matplotlib.pyplot 

from scipy import imresize, imread, imshow
from keras import models
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
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
            cpy = x.copy()
            struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size=morph_size)
            cpy = cv2.dilate(~cpy, struct, anchor=(-1, -1), iterations=1)
            x = ~cpy

            x = numpy.expand_dims(x, axis=4)

            data.append(x)

            labels.append(os.path.basename(folder))

data1 = numpy.asarray(data)
labels1 = numpy.asarray(labels)

x_train, x_test, y_train, y_test = train_test_split(data1, labels1, random_state=0, test_size=0.5)
x_train = numpy.array(x_train)
x_test = numpy.array(x_test)
y_train = numpy.array(y_train)
y_test = numpy.array(y_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

input_shape = (img_rows, img_cols, 1)

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

lb = LabelEncoder()

y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

with open(vars.label_obj_path, "w") as lb_obj:
    pickle.dump(lb, lb_obj)

model.save_weights(model_path)
print("Save model to disk")
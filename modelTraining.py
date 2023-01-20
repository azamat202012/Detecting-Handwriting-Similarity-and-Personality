from __future__ import print_function
# coding: utf-8

######################################################################################
# # TRAIN:
import os
import cv2
# simplified interface for building models
import keras
import pickle
import numpy as np
import matplotlib.pyplot as plt
# because our models are simple
from keras.models import Sequential
from keras.models import model_from_json
# for convolution (images) and pooling is a technique to help choose the most relevant features in an image
from keras.layers import Conv2D, MaxPooling2D
from sklearn.preprocessing import LabelEncoder
from scipy.misc import imresize
# dense means fully connected layers, dropout is a technique to improve convergence, flatten to reshape our matrices for feeding
# into respective layers
from keras.layers import Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split

# Load variables from another file
import variables as vars

img_rows, img_cols = vars.img_rows, vars.img_cols
batch_size = vars.batch_size
num_classes = vars.num_classes
epochs = vars.epochs
model_json_path = vars.model_json_path
model_path = vars.model_path
prediction_file_dir_path = vars.prediction_file_dir_path

# Directory containing the images
path = 'FEATURE-BASED-IMAGES/'

# Lists to store the data and labels
data = []
labels = []

# Iterate over all files in the directory
for folder, subfolders, files in os.walk(path):
  # Iterate over all images in the current folder
  for name in files:
    if name.endswith('.jpg'):
      # Read the image and convert it to grayscale
      image = cv2.imread(folder + '/' + name, cv2.IMREAD_GRAYSCALE)
      
      # Resize the image
      image = imresize(image, (img_rows, img_cols))

      # Threshold the image to convert it to a binary image
      _, image = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)

      # Dilate the image
      morph_size = (2, 2)
      image_copy = image.copy()
      struct = cv2.getStructuringElement(cv2.MORPH_RECT, morph_size)
      image_copy = cv2.dilate(~image_copy, struct, anchor=(-1, -1), iterations=1)
      image = ~image_copy

      # Add a fourth dimension to the image for Keras
      image = np.expand_dims(image, axis=4)

      # Add the image and its label to the data and labels lists
      data.append(image)
      labels.append(os.path.basename(folder))

# Convert the data and labels to numpy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, random_state=0, test_size=0.5)

# Convert the data to float32 and normalize it by dividing by 255
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Input shape for the model
input_shape = (img_rows, img_cols, 1)

# Build the model
model = Sequential()

# Add convolutional layers with ReLU activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add dropout layer to prevent overfitting
# Add dropout layer to prevent overfitting
model.add(Dropout(0.25))

# Flatten the data for the dense layers
model.add(Flatten())

# Add dense layers with ReLU activation
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

# Add output layer with softmax activation
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with Adam optimizer and categorical cross-entropy loss
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit the model to the training data
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

# Evaluate the model on the testing data
score = model.evaluate(x_test, y_test, verbose=0)

# Print the results
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Plot the training and testing accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot the training and testing loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model to a JSON file
model_json = model.to_json()
with open(model_json_path, "w") as json_file:
    json_file.write(model_json)

# Save the model weights to a file
model.save_weights(model_path)

# Make predictions on the test data
predictions = model.predict(x_test)

# Save the predictions to a file
prediction_file_path = prediction_file_dir_path + "prediction.pkl"
with open(prediction_file_path, "wb") as f:
  pickle.dump(predictions, f)

# Convert the labels to integers
label_encoder = LabelEncoder()
integer_encoded_y_test = label_encoder.fit_transform(y_test)

# Get the index of the class with the highest predicted probability for each sample
predictions_index = np.argmax(predictions, axis=1)

# Print the con# Make predictions on the test data
predictions = model.predict(x_test)

# Save the predictions to a file
prediction_file_path = prediction_file_dir_path + "prediction.pkl"
with open(prediction_file_path, "wb") as f:
  pickle.dump(predictions, f)

# Convert the labels to integers
label_encoder = LabelEncoder()
integer_encoded_y_test = label_encoder.fit_transform(y_test)

# Get the index of the class with the highest predicted probability for each sample
predictions_index = np.argmax(predictions, axis=1)
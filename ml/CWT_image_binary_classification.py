'''
Code by Dr Wei Li
24th Oct 2024
'''


# import libaries as needed

import os
import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

# Set image path and size

image_directory = r'J:/AlSi10Mg single layer ffc/CWT_labelled_01/'
SIZE_X = 100
SIZE_Y = 256
dataset = [] 
label = []

# Creat image and label lists
# Change the image type

NP_images = os.listdir(image_directory + '0/')
for i, image_name in enumerate(NP_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '0/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE_X, SIZE_Y))
        dataset.append(np.array(image))
        label.append(0)

P_images = os.listdir(image_directory + '1/')
for i, image_name in enumerate(P_images):
    if (image_name.split('.')[1] == 'png'):
        image = cv2.imread(image_directory + '1/' + image_name)
        image = Image.fromarray(image)
        image = image.resize((SIZE_X, SIZE_Y))
        dataset.append(np.array(image))
        label.append(1)
		
# Convert to np array

dataset = np.array(dataset)
label = np.array(label)

# Split the data set as needed

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.2, random_state = 0)

#Data normalization (0,1) to help convergence
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

#Creat CNN stucture for binary classification

model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(SIZE_Y, SIZE_X, 3)),
    layers.Conv2D(16, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#Use CSVlogger to record training process

csv_logger = CSVLogger('ml/CWT_image_binary_classification.csv', append=True)

# Setting of model training (optimizer, learning rate, etc)

# Learning rate
learning_rate = 0.0001

# Optimizer
Adam_optimizer = Adam(learning_rate=learning_rate)

# Model compile

model.compile(optimizer= Adam_optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])
			  
# Model training

history = model.fit(X_train, 
                         y_train, 
                         batch_size = 8, 
                         verbose = 1, 
                         epochs = 50,      
                         validation_data=(X_test,y_test),
                         shuffle = False,
                         callbacks=[csv_logger] 
                     )
					 
# Model save

model.save('ml/CWT_image_binary_classification.h5') 

plt.style.use('classic')

#Test the model on one image 

n=1
img = X_test[1]
plt.imshow(img)
plt.show()
input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)

print("The prediction for this image is: ", model.predict(input_img))
print("The actual label for this image is: ", y_test[n])

#Instead f checking for each image, we can evaluate the model on all test data
#for accuracy

#Check the accuracy of trained model

# load model
model = load_model('ml/CWT_image_binary_classification.h5')
_, acc = model.evaluate(X_test, y_test)

#Print the testing  accuracy

print("Accuracy = ", (acc * 100.0), "%")
# -*- coding: utf-8 -*-

#The code within this file has been derived from:
#https://medium.com/@sidathasiri/building-a-convolutional-neural-network-for-image-classification-with-tensorflow-f1f2f56bd83b
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop 

base_path = os.getcwd()+'/Images'

X_train_path = os.path.join(base_path,'Train')
y_test_path = os.path.join(base_path,'Test')

#add image augmentation to train data to improve accuracy
X_generator = ImageDataGenerator(1.0/255, 
                                 rotation_range=40, 
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
y_generator = ImageDataGenerator(1.0/255.)

TARGET_SIZE = (80,80)


generate_X_data = X_generator.flow_from_directory(X_train_path,batch_size=20,class_mode='binary',target_size = TARGET_SIZE)

generate_y_data = y_generator.flow_from_directory(y_test_path,batch_size=20,class_mode='binary',target_size = TARGET_SIZE)

CNN_Model = keras.Sequential([
    #the images have been resized to 80,80 and are RGB so need a channel of 3
    layers.Input(shape=(80,80,3)),
    layers.Conv2D(filters=16, kernel_size=(3,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(filters=32, kernel_size=(3,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(filters=64, kernel_size=(3,3)),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

#RMSprop can be used since the classification is binary
CNN_Model.compile(optimizer=RMSprop(lr=0.001),
              loss='binary_crossentropy',
              metrics = ['accuracy'])

print('\n')

CNN_Model.fit(generate_X_data,
                   validation_data=generate_y_data,
                    steps_per_epoch=42,
                    epochs=10,
                    validation_steps=16,
                    verbose=1)

test_loss, test_acc = CNN_Model.evaluate(generate_y_data, verbose=2)
print('\n Test accuracy:', test_acc)

CNN_Model.save("CNN_Image_Classification_Model.h5")

import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from keras.optimizers import Adam

train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)
## Dia chi data Train
train_generator = train_data_gen.flow_from_directory(
        '/content/drive/MyDrive/abc/archive.zip (Unzipped Files)/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')



##### ## Dia chi data test

# Preprocess all train images
validation_generator = validation_data_gen.flow_from_directory(
        '/content/drive/MyDrive/abc/archive.zip (Unzipped Files)/test',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
## model
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

cv2.ocl.setUseOpenCL(False)
opt = tf.keras.optimizers.Adam(0.0001, decay=1e-6)

emotion_model.compile(loss='categorical_crossentropy', optimizer= opt, metrics=['accuracy'])

history = emotion_model.fit(train_generator, batch_size=64, 
                    validation_data=validation_generator, epochs=50)
file_name = '/content/drive/MyDrive/abc/model9.h5'
emotion_model.save(file_name)
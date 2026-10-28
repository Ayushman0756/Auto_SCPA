import tensorflow as tf
import os
from tensorflow import keras    
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
import matplotlib.pyplot as plt

#generators
train_ds= keras.utils.image_dataset_from_directory(
    directory = r"C:\AI\CNN-Project\Normal_vs_polyps_train",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
)
validation_ds= keras.utils.image_dataset_from_directory(
    directory = r"C:\AI\CNN-Project\Normal_vs_polyps_validation",
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(224, 224),
)

model=Sequential()

# Conv1
model.add(Conv2D(
    96, kernel_size=(11,11),
    padding='valid',
    strides=4,
    activation='relu',
    input_shape=(224,224,3)
))

model.add(MaxPooling2D(
    pool_size=(3,3),
    strides=2,
    padding='valid'
))

# Conv2
model.add(Conv2D(
    256, kernel_size=(5,5),
    padding='same',
    strides=1,
    activation='relu'
))

model.add(MaxPooling2D(
    pool_size=(3,3),
    strides=2,
    padding='valid'
))

# Conv3
model.add(Conv2D(
    384, kernel_size=(3,3),
    padding='same',
    strides=1,
    activation='relu'
))

# Conv4
model.add(Conv2D(
    384, kernel_size=(3,3),
    padding='same',
    strides=1,
    activation='relu'
))

# Conv5
model.add(Conv2D(
    256, kernel_size=(3,3),
    padding='same',
    strides=1,
    activation='relu'
))

model.add(MaxPooling2D(
    pool_size=(3,3),
    strides=2,
    padding='valid'
))


model.add(Flatten())

model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(4096,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
history=model.fit(train_ds,epochs=100,validation_data=validation_ds)

plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='validation')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='validation')
plt.legend()
plt.show()
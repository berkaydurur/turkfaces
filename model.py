import cv2
import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, Conv2D, Activation, Dropout
import tensorflow.keras.backend as K
import keras
import pydot
from keras.models import Sequential, Model
from keras.models import load_model
from tensorflow.keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import MaxPool2D, ZeroPadding2D
from keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

train_path = '/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/train'  # TO DO buralara split folder ile gerekli dosyalar gelecek
test_path = '/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/val'
class_names = os.listdir(train_path)
class_names_test = os.listdir(test_path)

trainDirNum = 0
valDirNum = 0
for root, dirs, files in os.walk("/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/train"):
    for name in files:
        trainDirNum += 1

for root, dirs, files in os.walk("/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/val"):
    for name in files:
        valDirNum += 1

print('File numbers ', trainDirNum, ' ', valDirNum)
train_batchsize = 64
val_batchsize = 64
img_rows = 224
img_cols = 224

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory("/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/train",
                                                    target_size=(224, 224), batch_size=32, shuffle=True,
                                                    class_mode='categorical')
test_generator = test_datagen.flow_from_directory("/home/deeplab/Masaüstü/berkaydurur/facesFinalInput_v2/val",
                                                  target_size=(224, 224), batch_size=32, shuffle=False,
                                                  class_mode='categorical')


def VGG16():
    model = Sequential()
    model.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='vgg16'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(2786, activation='softmax'))
    return model


model = VGG16()
model.summary()

tf.keras.utils.plot_model(model, to_file="deneme.png", show_shapes=True)

Vgg16 = Model(inputs=model.input, outputs=model.get_layer('vgg16').output)

opt = SGD(learning_rate=0.001, momentum=0.9)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy", "categorical_crossentropy"])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20, min_delta=0, restore_best_weights=True)

mc = ModelCheckpoint('/home/deeplab/Masaüstü/berkaydurur/best_model_v1.h5', monitor='val_loss', mode='min',
                     save_best_only=True, verbose=1)

nb_train_samples = trainDirNum  # # bu iki satir degisken olacak len(train)
nb_validation_samples = valDirNum  # bu iki satir degisken olacak len(test)
epochs = 200
batch_size = 64

history = model.fit_generator(train_generator, validation_data=test_generator, epochs=epochs, verbose=1, callbacks=[mc],
                        validation_steps=nb_validation_samples)


print(history.history)

model.evaluate_generator(test_generator)



plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('acc1.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('loss1.png')
plt.show()

model_json = model.to_json()
with open("/home/deeplab/Masaüstü/berkaydurur/model.json", "w") as json_file:
    json_file.write(model_json)

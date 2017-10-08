from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import (Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D)
# from keras.utils import visualize_util
from keras.utils import np_utils
import json

# from keras.models import model_from_json

# input data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# model and data processing constants
batch_size = 128
nb_classes = 10
nb_epoch = 1  # 12 TODO: just for testing!

# input image dimensions
img_rows, img_cols = 28, 28

# number of convolutional filters to use
nb_filters = 32

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

# number of color channels (greyscale: 1, RGB: 3)
nb_channels = 1

# define model
model = Sequential()

model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(img_rows, img_cols, nb_channels), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# build model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

# visualize model
# visualize_util.plot(loaded_model, to_file='simple_image_classification_architecture.png', show_shapes=True)

# prepare input data to fit model requirements
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, nb_channels)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, nb_channels)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# print output to give notice about inputs
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# train model
model.fit(X_train, Y_train,
          batch_size=batch_size,
          epochs=nb_epoch, verbose=1,
          validation_data=(X_test, Y_test))

# save model to be reusable
saved_model = model.to_json()
with open('simple_image_classification_architecture.json', 'w') as outfile:
    json.dump(saved_model, outfile)

# also save its weights (to have a checkpoint where to proceed from)
model.save_weights('simple_image_classification_weights.h5')

# with this code, you could load and reuse the model and its weights
# Load architecture
# with open('simple_image_classification_architecture.json', 'r') as architecture_file:
# model_architecture = json.load(architecture_file)
#
#
# loaded_model = model_from_json(model_architecture)
#
# Load weights
# loaded_model.load_weights('simple_image_classification_weights.h5')

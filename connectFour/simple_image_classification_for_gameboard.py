from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D
from keras.utils import np_utils, vis_utils
from keras import backend as K
import json

# model and data processing constants
batch_size = 16
nb_epochs = 12
nb_classes = 7

# input data properties
img_height, img_width = 70, 70
img_size = (img_width, img_height)
train_directory = 'data/train'
test_directory = 'data/test'
nb_train_samples = 22544
nb_test_samples = 22544

# size of pooling area for max pooling
nb_pool = 2

# convolution kernel size
nb_conv = 3

# number of color channels (greyscale: 1, RGB: 3)
nb_channels = 3

# based on defined mode in ~/.keras/keras.json, pick correct format
if K.image_data_format() == 'channels_first':
    input_shape = (nb_channels, img_width, img_height)
else:
    input_shape = (img_width, img_height, nb_channels)

# # define model
# model = Sequential()
#
# model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=input_shape, padding='valid'))
# model.add(Activation('relu'))
#
# model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
# model.add(Activation('relu'))
# model.add(MaxPool2D(pool_size=(nb_pool, nb_pool)))
#
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

model = Sequential()
model.add(Conv2D(32, (nb_conv, nb_conv), input_shape=input_shape, padding='valid'))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, (nb_conv, nb_conv)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
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
# vis_utils.plot_model(model, to_file='simple_image_classification_architecture.png', show_shapes=True)

# save model to be reusable
model_json = model.to_json()
with open('simple_image_classification_fgb_architecture.json', 'w') as model_file:
    json.dump(model_json, model_file)

# input data generator for training (doing only rescaling of color-values)
train_imagegen = ImageDataGenerator(rescale=1. / 255)
train_datagenerator = train_imagegen.flow_from_directory(train_directory, target_size=img_size, batch_size=batch_size)

# input data generator for testing (doing only rescaling of color-values)
test_imagegen = ImageDataGenerator(rescale=1. / 255)
test_datagenerator = test_imagegen.flow_from_directory(train_directory, target_size=img_size, batch_size=batch_size)

# load existing weights
model.load_weights('simple_image_classification_fgb_weights_run3.h5')

# train model
model.fit_generator(train_datagenerator,
                    steps_per_epoch=nb_train_samples // batch_size,
                    epochs=nb_epochs,
                    validation_data=test_datagenerator,
                    validation_steps=nb_test_samples // batch_size)

# save model weights (to have a checkpoint where to proceed from)
model.save_weights('simple_image_classification_fgb_weights_run4.h5')

# with this code, you could load and reuse the model and its weights
# from keras.models import model_from_json
#
# Load architecture
# with open('simple_image_classification_architecture.json', 'r') as architecture_file:
# model_architecture = json.load(architecture_file)
#
#
# loaded_model = model_from_json(model_architecture)
#
# Load weights
# loaded_model.load_weights('simple_image_classification_weights.h5')

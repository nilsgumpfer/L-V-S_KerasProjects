from keras.applications.vgg16 import VGG16
model = VGG16()

from quiver_engine import server
server.launch(model)
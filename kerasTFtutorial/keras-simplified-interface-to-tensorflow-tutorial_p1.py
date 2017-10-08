# import tensorflow and keras
import tensorflow as tf
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from tensorflow.examples.tutorials.mnist import input_data
from keras.metrics import categorical_accuracy as accuracy

# create tensorflow session
sess = tf.Session()

# register tf session with keras (it uses the session to initialize variables and stuff)
K.set_session(sess)

# this placeholder will contain our input digits, as flat vectors
img = tf.placeholder(tf.float32, shape=(None, 784))

# keras layers can be called on TensorFlow tensors:
x = Dense(128, activation='relu')(img)  # fully-connected layer with 128 units and ReLU activation
x = Dense(128, activation='relu')(x)
preds = Dense(10, activation='softmax')(x)  # output layer with 10 units and a softmax activation

# placeholder for labels and loss function
labels = tf.placeholder(tf.float32, shape=(None, 10))
loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

# this is the input data-set, containing vectorized image-data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

# train the model with a tf optimizer to minimize the results of the loss function (0.5: learning rate)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

# initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

# run training loop 100 times, using training data
with sess.as_default():
    for i in range(100):
        batch = mnist_data.train.next_batch(50)
        train_step.run(feed_dict={img: batch[0], labels: batch[1]})
        print("training")

# finally, evaluate what has been done by comparing outputs to defined values of test-data
acc_value = accuracy(labels, preds)
with sess.as_default():
    print(acc_value.eval(feed_dict={img: mnist_data.test.images, labels: mnist_data.test.labels}))
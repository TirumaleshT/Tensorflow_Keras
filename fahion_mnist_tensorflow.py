import tensorflow as tf
from tensorflow import keras
import numpy as np

#Load dataset
fashion_mnist = keras.datasets.fashion_mnist
(x_train, y_)train), (x_test, y_test) = fashion_mnist.load_data()

#scaling data between 0 and 1. diving x_train adn x_test by 255 coz the max pixel intensity of the images is 255.
x_train = x_train / 255
x_test = x_test / 255


#Define model
model = tf.keras.Sequential([keras.layers.Flatten(input_shape = (28,28)),
							keras.layers.Dense(128, activation = tf.nn.relu),
							keras.layers.Dense(10, activation = tf.nn.softmax)])
							
#compile the model
model.compile(loss = 'sparse_categorical_crossentropy',
			optimizer = 'tf.train.AdamOptimizer()',
			metrics = ['accuracy'])
			
#train model
model.fit(x_train, y_train)

#calculate loss and accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test)

#make a predicton. predict will return the probabilities of the total output neurons
predictions = model.predict(x_test[:10])

#print max value for each prediction
[print(np.argmax(predictions[i])) for i in range(len(predictions))]
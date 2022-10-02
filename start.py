import tensorflow as tf
print(tf.__version__)

import numpy as np

print(f"model na podstawie funkcji liniowej: 2x-1")

X = np.array([-1,0,1,2,3,4])
y = np.array([-3,-1,1,3,5,7])



from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss="mean_squared_error")
model.fit(X,y, epochs=500)

print(model.predict([-1,1,4,10]))
print("="*50)

# model mnist

from tensorflow.keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network = Sequential()
network.add(Dense(512, activation='relu', input_shape=(28*28,)))
network.add(Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['accuracy'])

network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")


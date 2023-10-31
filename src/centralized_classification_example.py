import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from keras.models import load_model

### Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train [0:512]
y_train = y_train [0:512]

### Filter data
train_mask_0_1 = np.isin(y_train, [0, 1])
x_train, y_train = x_train [train_mask_0_1], y_train [train_mask_0_1]

### Resize data
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [32,32])


### Load/create, train and save model
try:
  model = load_model ('mobilenetv2_mnist.keras')
except:
  model = tf.keras.applications.MobileNetV2((32,32,1), classes=10, weights=None)
model.compile ("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
history = model.fit (x_train, y_train, epochs=25, batch_size=64, validation_split=0.2, verbose=2, workers=1, use_multiprocessing=True)
model.save('mobilenetv2_mnist.keras')


### Evaluate model
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [32,32])
model.evaluate (x_test, y_test, verbose=2)

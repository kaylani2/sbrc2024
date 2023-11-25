import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense

def load_compiled_model (model='MobileNetV2'):
  print ('Loading model...')

  if (model == 'MobileNetV2'):
    model = tf.keras.applications.MobileNetV2(
      (32,32,1),
      classes=10,
      alpha=0.4,
      weights=None)

    optimizer = Adam(learning_rate=1e-2)

    model.compile (
      loss="sparse_categorical_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"])


  if (model == 'custom'):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(32,32,1)))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=3,activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    optimizer = Adam(learning_rate=1e-3)

    model.compile (
      loss="sparse_categorical_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"])

  print(model.summary())
  return model


def load_dataset (dataset='mnist', resize=True):
  print ('Loading data...')

  if (dataset=='mnist'):
    try:
      (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    except:
      path = '/home/gta/.keras/datasets/mnist.npz'
      with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test   = f['x_test'], f['y_test']


  if (dataset=='cifar10'):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    # Normalize pixel values to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0


  if (resize):
    print ('Resizing data...')
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = tf.image.resize(x_train, [32,32])
    x_test = np.expand_dims(x_test, axis=-1)
    x_test = tf.image.resize(x_test, [32,32])

  return (x_train, y_train), (x_test, y_test)

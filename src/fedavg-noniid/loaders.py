import tensorflow as tf
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, Flatten, Dense

def load_compiled_model (model='MobileNetV2'):
  print ('Loading model...')

  if (model == 'MobileNetV2'):
    model = tf.keras.applications.MobileNetV2(
      (32,32,1),
      classes=10,
      alpha=0.7,
      weights=None)

    optimizer = Adam(learning_rate=1e-3)

    model.compile (
      loss="sparse_categorical_accuracy",
      optimizer=optimizer,
      metrics=["accuracy"])


  if (model == 'custom'):
    model = Sequential()

    model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
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

    optimizer = Adam(learning_rate=1e-2)

    model.compile (
      loss="sparse_categorical_crossentropy",
      optimizer=optimizer,
      metrics=["accuracy"])

  return model

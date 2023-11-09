import flwr as fl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from keras.models import load_model
from keras.optimizers import Adam

batchsizes=[64, 128, 256]
learning_rates=[1e-2, 1e-4]
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
METRICS = ["accuracy"]


### Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train [0:2048]
#y_train = y_train [0:2048]

### Filter data
#train_mask_0_1 = np.isin(y_train, [0, 1])
#x_train, y_train = x_train [train_mask_0_1], y_train [train_mask_0_1]

### Resize data
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [32,32])
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [32,32])

for BATCH_SIZE in batchsizes:
  for LEARNING_RATE in learning_rates:
    print ('Batch size:', BATCH_SIZE)
    print ('Learning rate:', LEARNING_RATE)

    ### Load/create, train and save model
    try:
      model = load_model ('mobilenetv2_mnist.keras')
    except:
      model = tf.keras.applications.MobileNetV2((32,32,1), classes=10, weights=None)
    optimizer = Adam(learning_rate = LEARNING_RATE)
    model.compile (loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=METRICS)
    history = model.fit (x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, verbose=2, workers=1, use_multiprocessing=True)
    #model.save('mobilenetv2_mnist.keras')


    ### Evaluate model
    results = model.evaluate (x_test, y_test, verbose=2)
    print ("Test loss, test accuracy:", results)

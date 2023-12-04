import tensorflow as tf
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam

batch_sizes=[64, 128, 256]
learning_rates=[1e-2, 1e-4]
NUM_EPOCHS = 10
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

for batch_size in batch_sizes:
  for learning_rate in learning_rates:
    print ('Batch size:', batch_size)
    print ('Learning rate:', learning_rate)

    ### Load/create, train and save model
    try:
      model = load_model ('mobilenetv2_mnist.keras')
    except:
      model = tf.keras.applications.MobileNetV2((32,32,1), classes=10, weights=None)
    optimizer = Adam(learning_rate = learning_rate)
    model.compile (loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=METRICS)
    print ('Running...')
    history = model.fit (x_train, y_train, epochs=NUM_EPOCHS, batch_size=batch_size, validation_split=0.2, verbose=2, workers=1, use_multiprocessing=True)
    #model.save('mobilenetv2_mnist.keras') ### K: Don't save model to generate four examples trained from scratch.


    ### Evaluate model
    results = model.evaluate (x_test, y_test, verbose=2)
    print ("Test loss, test accuracy:", results)

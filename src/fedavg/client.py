import flwr as fl
import tensorflow as tf
import numpy as np
from keras.optimizers import Adam
from sys import argv
from logging import INFO, DEBUG
from flwr.common.logger import log

LEARNING_RATE=1e-2
NUM_EPOCHS=1
BATCH_SIZE=64
METRICS = ["accuracy"]
VERBOSE=2
client_index = argv[1]

### Setup logging
log_file = "client_main_"+str(client_index)+".log"
fl.common.logger.configure(identifier="mestrado", filename=log_file)

### Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

### Resize data
x_train = np.expand_dims(x_train, axis=-1)
x_train = tf.image.resize(x_train, [32,32])
x_test = np.expand_dims(x_test, axis=-1)
x_test = tf.image.resize(x_test, [32,32])

### Define model
model = tf.keras.applications.MobileNetV2((32,32,1), classes=10, weights=None)
optimizer = Adam(learning_rate=LEARNING_RATE)
model.compile (loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=METRICS)

class MNISTClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    log(DEBUG, f"Client {client_index} is doing fit() with config: {config}")
    model.set_weights(parameters)
    model.fit (x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, use_multiprocessing=True)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    log(DEBUG, f"Client {client_index} is doing evaluate() with verbose: {VERBOSE}")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
    log(INFO, f"Client {client_index} achieved loss: {loss} and accuracy: {accuracy}")
    return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="[::]:8080", client=MNISTClient())

import flwr as fl
import tensorflow as tf
import numpy as np
import sys
from keras.optimizers import Adam
from sys import argv
from logging import INFO, DEBUG
from flwr.common.logger import log

if len(sys.argv) > 2:
  num_clients = int(argv[1])
  client_index = int(argv[2])
  print("num_clients:", num_clients)
  print("client_index:", client_index)
else:
  print ("Usage: python client.py num_clients client_index")
  sys.exit()

LEARNING_RATE=1e-2
NUM_EPOCHS=1
BATCH_SIZE=64
METRICS = ["accuracy"]
VERBOSE=2
current_round=0

### Setup logging
filename = "client_main_"+str(client_index).zfill(len(str(num_clients)))+"_"+str(num_clients)+"_clients.log"
fl.common.logger.configure(identifier="mestrado", filename=filename)

### Load data
try:
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
except:
  path = '/home/gta/.keras/datasets/mnist.npz'
  with np.load(path, allow_pickle=True) as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test   = f['x_test'], f['y_test']

### Split data (two random labels for each client, except for five clients)
samples=None
if (num_clients == 5): ### Must ensure all labels are present
  samples = str([2 * (client_index - 1), 2 * (client_index - 1) + 1]) ### K: Thanks, ChatGPT.
  data_mask = np.isin(y_train, [2 * (client_index - 1), 2 * (client_index - 1) + 1]) ### K: Thanks, ChatGPT.
  x_train, y_train = x_train [data_mask], y_train [data_mask]
  x_test, y_test = x_test [data_mask], y_test [data_mask]
else:
  pass ### TODO: choose between random overlapping ou non-overlapping labels

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
    log(DEBUG, f"Client {client_index} is doing fit() with config: {config} and samples={samples}")
    current_round=config['server_round']
    print('current_round:', current_round)
    model.set_weights(parameters)
    model.fit (x_train, y_train, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, use_multiprocessing=True)
    return model.get_weights(), len(x_train), {}

  def evaluate(self, parameters, config):
    log(DEBUG, f"Client {client_index} is doing evaluate() with verbose: {VERBOSE}")
    model.set_weights(parameters)
    loss, accuracy = model.evaluate(x_test, y_test, verbose=VERBOSE)
    log(INFO, f"Client {client_index} achieved loss={loss} and accuracy={accuracy} on round: {current_round} with config: {config}")
    return loss, len(x_test), {"accuracy": float(accuracy)}

fl.client.start_numpy_client(server_address="[::]:8080", client=MNISTClient())
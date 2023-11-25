import flwr as fl
import tensorflow as tf
import numpy as np
import sys
from keras.optimizers import Adam
from sys import argv
from logging import INFO, DEBUG
from flwr.common.logger import log
from loaders import load_compiled_model, load_dataset

if len(sys.argv) > 2:
  num_clients = int(argv[1])
  client_index = int(argv[2])
  print("num_clients:", num_clients)
  print("client_index:", client_index)
else:
  print ("Usage: python client.py num_clients client_index")
  sys.exit()


DATASET='cifar10'
RESIZE=False
MODEL='custom_cifar10'
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
(x_train, y_train), (x_test, y_test) = load_dataset(DATASET, resize=RESIZE)

### Split data (clients must not have the same samples)
# Split train
subset_size = len(x_train) // num_clients
x = [x_train[i*subset_size: (i+1)*subset_size] for i in range(num_clients)]
y = [y_train[i*subset_size: (i+1)*subset_size] for i in range(num_clients)]
x_train = x[client_index-1] ### Each client receives data related to its index
y_train = y[client_index-1]
# Split test
subset_size = len(x_test) // num_clients
x = [x_test[i*subset_size: (i+1)*subset_size] for i in range(num_clients)]
y = [y_test[i*subset_size: (i+1)*subset_size] for i in range(num_clients)]
x_test = x[client_index-1] ### Each client receives data related to its index
y_test = y[client_index-1]
del x
del y

### Define and load model
model = load_compiled_model(MODEL)

class MNISTClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return model.get_weights()

  def fit(self, parameters, config):
    log(DEBUG, f"Client {client_index} is doing fit() with config: {config}")
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

fl.client.start_numpy_client(server_address="[::]:50077", client=MNISTClient())

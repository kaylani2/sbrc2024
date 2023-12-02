import sys
import flwr as fl
import tensorflow as tf
import numpy as np
from sys import argv
from keras.optimizers import Adam
from flwr.common.logger import log
from logging import INFO, DEBUG
from loaders import load_compiled_model, load_dataset

if (len(sys.argv) > 4):
  num_clients = int(argv[1])
  client_index = int(argv[2])
  MODEL = str(argv[3])
  DATASET = str(argv[4])
  print("num_clients:", num_clients)
  print("client_index:", client_index)
  print("model:", MODEL)
  print("dataset:", DATASET)
else:
  print ("Usage: python client.py num_clients client_index model dataset")
  sys.exit()

RESIZE=True
LEARNING_RATE=1e-2
NUM_EPOCHS=1
BATCH_SIZE=64
METRICS = ["accuracy"]
VERBOSE=2
current_round=0

### Setup logging
filename = "client_main_"+str(client_index).zfill(len(str(num_clients)))+"_"+str(num_clients)+"_clients.log"
fl.common.logger.configure(identifier="mestrado", filename=filename)

#### Load data
(x_train, y_train), (x_test, y_test) = load_dataset(DATASET, RESIZE)

### Split data
print ('Splitting data...')
samples=None
fake_index = (client_index - 1) % 5 + 1 
samples = str([2 * (fake_index - 1), 2 * (fake_index - 1) + 1]) ### K: Thanks, ChatGPT.
### Split train
train_mask = np.isin(y_train, [2 * (fake_index - 1), 2 * (fake_index - 1) + 1])
x_train, y_train = x_train [train_mask], y_train [train_mask]
### Split test
test_mask = np.isin(y_test, [2 * (fake_index - 1), 2 * (fake_index - 1) + 1])
x_test, y_test = x_test [test_mask], y_test [test_mask]

if (client_index <= 5):
  x_train, y_train = x_train [:len(x_train)//5], y_train [:len(y_train)//5]
elif (client_index <= 10):
  x_train, y_train = x_train [len(x_train)//5 : 2*len(x_train)//5], y_train [len(y_train)//5 : 2*len(y_train)//5]
elif (client_index <= 15):
  x_train, y_train = x_train [2*len(x_train)//5 : 3*len(x_train)//5], y_train [2*len(y_train)//5 : 3*len(y_train)//5]
elif (client_index <= 20):
  x_train, y_train = x_train [3*len(x_train)//5 : 4*len(x_train)//5], y_train [3*len(y_train)//5 : 4*len(y_train)//5]
else:
  x_train, y_train = x_train [4*len(x_train)//5:], y_train [4*len(y_train)//5:]

print (f"client_index: {client_index}")
print (f"samples: {samples}")
print (f"len(x_train): {len(x_train)}")
print (f"len(x_test): {len(x_test)}")
print (f"train_mask: {train_mask}")
print (f"test_mask:  {test_mask}")
print (f"y_train: {y_train [0:35]}")
print (f"y_test:  {y_test  [0:35]}")

### Define and load model
model = load_compiled_model(MODEL)

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

fl.client.start_numpy_client(server_address="172.31.46.102:50077", client=MNISTClient())

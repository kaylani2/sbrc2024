import sys
import flwr as fl
import tensorflow as tf
import numpy as np
from sys import argv
from keras.optimizers import Adam
from flwr.common.logger import log
from logging import INFO, DEBUG
from loaders import load_compiled_model, load_dataset
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage

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

if (client_index <= 6):
  # Create an ImageDataGenerator for augmentation
  datagen = ImageDataGenerator(
      rotation_range=10,  # Rotate images by up to 10 degrees
      width_shift_range=0.1,  # Shift images horizontally by 10% of the width
      height_shift_range=0.1,  # Shift images vertically by 10% of the height
      zoom_range=0.1  # Zoom in/out by 10%
  )
  # Fit the ImageDataGenerator to the data
  datagen.fit(x_train)

  # Generate augmented samples and append to x_train and y_train
  augmented_samples = len(x_train)  # Number of augmented samples to generate
  augmented_x = []
  augmented_y = []

  for x_batch, y_batch in datagen.flow(x_train, y_train, batch_size=1):
      augmented_x.append(x_batch)
      augmented_y.append(y_batch)
      if len(augmented_x) >= augmented_samples:
          break

  if (DATASET == 'mnist'):
    # Convert lists to numpy arrays and concatenate with original data
    augmented_x = np.array(augmented_x).reshape(-1, 32, 32, 1)
    augmented_y = np.array(augmented_y).reshape(-1)

  elif (DATASET == 'cifar10'):
    # Convert lists to numpy arrays and concatenate with original data
    augmented_x = np.array(augmented_x)
    augmented_y = np.array(augmented_y)
    # Reshape augmented_x to match CIFAR-10 image shape
    augmented_x = augmented_x.reshape(-1, 32, 32, 3)
    augmented_y = augmented_y.reshape(-1)
    
  x_train = np.concatenate((x_train, augmented_x), axis=0)
  y_train = np.concatenate((y_train, augmented_y), axis=0)
  # Check the shape of augmented data
  print("New x_train shape:", x_train.shape)
  del augmented_x
  del augmented_y
  #print("Augmented x_train shape:", x_train_augmented.shape)

#if (client_index <= 6):
#  x_train, y_train = x_train [:2*len(x_train)//6], y_train [:2*len(y_train)//6]
#elif (client_index <= 10):
#  x_train, y_train = x_train [2*len(x_train)//6 : 3*len(x_train)//6], y_train [2*len(y_train)//6 : 3*len(y_train)//6]
#elif (client_index <= 15):
#  x_train, y_train = x_train [3*len(x_train)//6 : 4*len(x_train)//6], y_train [3*len(y_train)//6 : 4*len(y_train)//6]
#elif (client_index <= 20):
#  x_train, y_train = x_train [4*len(x_train)//6 : 5*len(x_train)//6], y_train [4*len(y_train)//6 : 5*len(y_train)//6]
#else:
#  x_train, y_train = x_train [5*len(x_train)//6:], y_train [5*len(y_train)//6:]

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

fl.client.start_numpy_client(server_address="[::]:50077", client=MNISTClient())

import sys
import flwr as fl
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from flwr.common import NDArrays, Scalar
from typing import Dict, Optional, Tuple, List, Union
from sys import argv
from loaders import load_compiled_model, load_dataset

if len(sys.argv) > 2:
  num_clients = int(argv[1])
  num_rounds = int(argv[2])
else:
  print ("Usage: python server.py num_clients num_rounds")
  sys.exit()

DATASET='cifar10'
RESIZE=False
MODEL='custom_cifar10'
MIN_FIT_CLIENTS=num_clients
MIN_AVAILABLE_CLIENTS=num_clients
FRACTION_FIT=1.0
ROUND_TIMEOUT=None
SERVER_ADDRESS="0.0.0.0:50077"

### Setup logging.
filename="server_main_"+str(num_rounds)+"rounds_"+str(num_clients)+"clients_fedavg.log"
fl.common.logger.configure(identifier="mestrado", filename=filename)

### Define and load model
model = load_compiled_model(MODEL)

### K: Used for centralized evaluation.
def get_evaluate_fn(model):
  """Return an evaluation function for server-side evaluation."""

  (_, _), (x_test, y_test) = load_dataset(DATASET, resize=RESIZE)

  # The `evaluate` function will be called after every round
  def evaluate(
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
  ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    model.set_weights(parameters)  # Update model with the latest parameters
    loss, accuracy = model.evaluate(x_test, y_test)
    return loss, {"accuracy": accuracy}

  return evaluate

### K: Used to output the round number on each client log. This makes plotting the results easier.
def fit_round(server_round: int) -> Dict:
  """Send round number to client."""
  return {"server_round": server_round}

def evaluate_config(server_round: int) -> Dict:
  """Send round number to client."""
  return {"server_round": server_round}


class SaveModelStrategy(fl.server.strategy.FedAvg):
  def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
    failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:
    # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
    aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
    if aggregated_parameters is not None:
      # Convert `Parameters` to `List[np.ndarray]`
      aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
      # Save aggregated_ndarrays
      print(f"Saving round {server_round} aggregated_ndarrays...")
      model_filename=("model_round-"+str(server_round).zfill(len(str(num_rounds)))+"-"+str(num_clients)+"clients"+str(num_rounds)+"rounds-weights.npz")
      np.savez(f"model_round-{server_round}-{num_clients}clients-weights.npz", *aggregated_ndarrays)

    return aggregated_parameters, aggregated_metrics

fl.server.start_server(
  server_address=SERVER_ADDRESS,
  config=fl.server.ServerConfig(
    num_rounds=num_rounds,
    round_timeout=ROUND_TIMEOUT,
  ),
  #strategy=fl.server.strategy.FedAvg(
  strategy=SaveModelStrategy(
    fraction_fit=FRACTION_FIT,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    on_fit_config_fn=fit_round,
    on_evaluate_config_fn=evaluate_config,
    evaluate_fn=get_evaluate_fn(model)
  )
)

### K: Com PyTorch...
#To load your progress, you simply append the following lines to your code. Note that this will iterate over all saved checkpoints and load the latest one:
#
#net = cifar.Net().to(DEVICE)
#list_of_files = [fname for fname in glob.glob("./model_round_*")]
#latest_round_file = max(list_of_files, key=os.path.getctime)
#print("Loading pre-trained model from: ", latest_round_file)
#state_dict = torch.load(latest_round_file)
#net.load_state_dict(state_dict)

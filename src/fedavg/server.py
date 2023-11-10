import flwr as fl
from typing import Dict

### Setup logging.
fl.common.logger.configure(identifier="mestrado", filename="server_main.log")

NUM_ROUNDS=50
MIN_FIT_CLIENTS=5
MIN_AVAILABLE_CLIENTS=5
FRACTION_FIT=1.0
ROUND_TIMEOUT=None
SERVER_ADDRESS="0.0.0.0:8080"

### K: Used to output the round number on each client log. This makes plotting the results easier.
def fit_round(server_round: int) -> Dict:
  """Send round number to client."""
  return {"server_round": server_round}

fl.server.start_server(
  server_address=SERVER_ADDRESS,
  config=fl.server.ServerConfig(
    num_rounds=NUM_ROUNDS,
    round_timeout=ROUND_TIMEOUT,
  ),
  strategy=fl.server.strategy.FedAvg(
    fraction_fit=FRACTION_FIT,
    min_fit_clients=MIN_FIT_CLIENTS,
    min_available_clients=MIN_AVAILABLE_CLIENTS,
    on_fit_config_fn=fit_round
  )
)

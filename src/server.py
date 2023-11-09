import flwr as fl
NUM_ROUNDS=3

fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
  strategy=fl.server.strategy.FedAvg(
    min_fit_clients=2,
    min_available_clients=2
  )
)

#!/bin/bash

declare -u num_clients=10
declare -u num_rounds=10

python server.py $num_clients $num_rounds > server-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
for i in `eval echo {1..$num_clients}`
do
  python client.py $num_clients $i > client$i-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
done

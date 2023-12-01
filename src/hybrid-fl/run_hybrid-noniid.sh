#!/bin/bash

declare -u num_clients=25
declare -u num_rounds=500
model="custom"
dataset="mnist"

python server.py $num_clients $num_rounds $model $dataset > server-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
sleep 5
for i in `eval echo {1..$num_clients}`
do
  python client.py $num_clients $i $model $dataset > client$i-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
done

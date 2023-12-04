#!/bin/bash

declare -u num_clients=6
declare -u num_rounds=500
model="custom_cifar10"
dataset="cifar10"

python server.py $num_clients $num_rounds $model $dataset > server-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
sleep 5
for i in `eval echo {1..$num_clients}`
do
  python client25percent.py $num_clients $i $model $dataset > client$i-$num_rounds-rounds-$num_clients-clients.bash.log 2>&1 &
done

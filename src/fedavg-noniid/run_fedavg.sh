#!/bin/bash

declare -u num_clients=2

python server.py $num_clients > server.bash.log 2>&1 &
for i in `eval echo {1..$num_clients}`
do
  python client.py $num_clients $i > client$i.bash.log 2>&1 &
done

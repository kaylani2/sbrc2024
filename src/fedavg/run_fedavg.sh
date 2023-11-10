#!/bin/bash

python server.py > server.log 2>&1 &
for i in {1..2} # define number of clients here
do
  python client.py $i > client$i.log 2>&1 &
done

#!/bin/bash

### For running the centralized example
#python centralized_classification_example.py > centralized.log 2>&1 &

python server.py > server.log 2>&1 &
for i in {1..3} # define number of clients here
do
  python client.py $i > client$i.log 2>&1 &
done

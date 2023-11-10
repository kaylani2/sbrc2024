#!/bin/bash

python server.py 3 > server.log 2>&1 &
for i in {1..3} # define number of clients here
do
  python client.py 3 $i > client$i.log 2>&1 &
done

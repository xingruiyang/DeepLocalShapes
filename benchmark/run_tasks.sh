#!/bin/bash

./benchmark/train.sh examples/table.stl output/table_train 32 0.1 1
./benchmark/train.sh examples/table.stl output/table_train2 32 0.3 1
./benchmark/train.sh examples/table.stl output/table_train 16 0.1 1
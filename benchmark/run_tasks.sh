#!/bin/bash

# Train the network on random shapes

echo "Generating random shapes...";

if [ ! -f examples/train.ply ]; then
    python3 samplers/random_mesh.py \
        --num_shapes 30 \
        --export examples/train.ply;
fi

if [ ! -f examples/eval.ply ]; then
    python3 samplers/random_mesh.py \
        --num_shapes 30 \
        --export examples/eval.ply;
fi

echo "Train on random shapes...";
./benchmark/train.sh examples/train.ply output/125/train 125 0.1
./benchmark/train.sh examples/train.ply output/64/train 64 0.1
./benchmark/train.sh examples/train.ply output/32/train 32 0.1

echo "Evaluate on random shapes...";
./benchmark/eval.sh examples/eval.ply output/125/eval output/125/train 125 0.1
./benchmark/eval.sh examples/eval.ply output/64/eval output/64/train 64 0.1
./benchmark/eval.sh examples/eval.ply output/32/eval output/32/train 32 0.1


echo "Evaluate on chairs...";
./benchmark/eval.sh examples/chair.stl output/125/chair output/125/train 125 0.1 1
./benchmark/eval.sh examples/chair.stl output/64/chair output/64/train 64 0.1 1
./benchmark/eval.sh examples/chair.stl output/32/chair output/32/train 32 0.1 1

echo "Evaluate on tables...";
./benchmark/eval.sh examples/table.stl output/125/table output/125/train 125 0.1 1
./benchmark/eval.sh examples/table.stl output/64/table output/64/train 64 0.1 1
./benchmark/eval.sh examples/table.stl output/32/table output/32/train 32 0.1 1
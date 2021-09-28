#!/bin/bash

# echo "Generating random shapes...";
# python3 samplers/random_samples.py  output/rand_train --num-shapes 225 --network models/transformer.pth;
# python3 samplers/random_samples.py  output/rand_eval --num-shapes 25 --network models/transformer.pth;
# if [ ! -f examples/train.ply ]; then
#     python3 samplers/random_mesh.py \
#         --num_shapes 30 \
#         --export examples/train.ply;
# fi

# if [ ! -f examples/eval.ply ]; then
#     python3 samplers/random_mesh.py \
#         --num_shapes 30 \
#         --export examples/eval.ply;
# fi

# echo "Train on random shapes...";
# ./benchmark/train.sh examples/train.ply output/125/train 125 0.1
# ./benchmark/train.sh examples/train.ply output/64/train 64 0.1
# ./benchmark/train.sh examples/train.ply output/32/train 32 0.1

# echo "Evaluate on random shapes...";
# ./benchmark/eval.sh examples/eval.ply output/125/eval output/125/train 125 0.1
# ./benchmark/eval.sh examples/eval.ply output/64/eval output/64/train 64 0.1
# ./benchmark/eval.sh examples/eval.ply output/32/eval output/32/train 32 0.1


# echo "Evaluate on chairs...";
# ./benchmark/eval.sh examples/chair.stl output/125/chair output/125/train 125 0.1 1
# ./benchmark/eval.sh examples/chair.stl output/64/chair output/64/train 64 0.1 1
# ./benchmark/eval.sh examples/chair.stl output/32/chair output/32/train 32 0.1 1

# echo "Evaluate on tables...";
# ./benchmark/eval.sh examples/table.stl output/125/table output/125/train 125 0.1 1
# ./benchmark/eval.sh examples/table.stl output/64/table output/64/train 64 0.1 1
# ./benchmark/eval.sh examples/table.stl output/32/table output/32/train 32 0.1 1

# python3 samplers/random_samples.py  output/train --num-shapes 225 --network models/transformer.pth;
# python3 samplers/random_samples.py  output/eval --num-shapes 25 --network models/transformer.pth;

# echo "Train on random shapes...";
./benchmark/train.sh output/train output/train/125 125 -1
./benchmark/train.sh output/train output/train/64 64 -1
./benchmark/train.sh output/train output/train/32 32 -1

# echo "Evaluate on random shapes...";
./benchmark/eval.sh output/eval output/eval/125 output/train/125 125 -1
./benchmark/eval.sh output/eval output/eval/64 output/train/64 64 -1
./benchmark/eval.sh output/eval output/eval/32 output/train/32 32 -1

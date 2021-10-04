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
# ./benchmark/eval.sh examples/chair.stl output/chair/125 output/train/125 125 -1 1
# ./benchmark/eval.sh examples/chair.stl output/chair/64 output/train/64 64 -1 1
# ./benchmark/eval.sh examples/chair.stl output/chair/32 output/train/32 32 -1 1

# echo "Evaluate on tables...";
# ./benchmark/eval.sh examples/table.stl output/125/table output/125/train 125 0.1 1
# ./benchmark/eval.sh examples/table.stl output/64/table output/64/train 64 0.1 1
# ./benchmark/eval.sh examples/table.stl output/32/table output/32/train 32 0.1 1

# python3 samplers/random_samples.py output/train --num-shapes 196 --network models/transformer.pth;
# python3 samplers/random_samples.py output/eval --num-shapes 16 --network models/transformer.pth;

# echo "Train on random shapes...";
# ./benchmark/train.sh output/train output/train/125 125 -1
# ./benchmark/train.sh output/train output/train/64 64 -1
# ./benchmark/train.sh output/train output/train/32 32 -1

# echo "Evaluate on random shapes...";
# ./benchmark/eval.sh output/eval output/eval/125 output/train/125 125 -1
# ./benchmark/eval.sh output/eval output/eval/64 output/train/64 64 0.3
# ./benchmark/eval.sh output/eval output/eval/32 output/train/32 32 0.3

# ./benchmark/sample_and_train.sh examples/chair.stl output/chair 64 0.1 1
# ./benchmark/sample_and_eval.sh examples/chair.stl output/chair/eval output/chair 64 0.1 1

# python3 samplers/sample_3dwarehouse.py \
    # dataset/3DWareHouse/ \
    # output/shapenet-train \
    # --voxel_size 0.03 \
    # --network models/transformer.pth 
# ./benchmark/shape_net_and_train.sh output/shapenet-train output/shapenet-train 125 -1
# ./benchmark/sample_and_eval.sh examples/chair.stl output/chair output/shapenet-train 125 -1 1

for id in {0..19}; do
./benchmark/shape_net_and_train.sh output/shape-samples/sofa/$id output/shape-logs/sofa/$id 125 -1
done

for id in {0..19}; do
./benchmark/shape_net_and_train.sh output/shape-samples/airliner/$id output/shape-logs/airliner/$id 125 -1
done

for id in {0..19}; do
./benchmark/shape_net_and_train.sh output/shape-samples/lamp/$id output/shape-logs/lamp/$id 125 -1
done

for id in {0..19}; do
./benchmark/shape_net_and_train.sh output/shape-samples/chair/$id output/shape-logs/chair/$id 125 -1
done

for id in {0..19}; do
./benchmark/shape_net_and_train.sh output/shape-samples/table/$id output/shape-logs/table/$id 125 -1
done
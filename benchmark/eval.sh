#!/usr/bin/bash

input=$1
output=$2
network=$3
LATENT_SIZE=$4
CLAMP_DIST=$5
VOXLE_SIZE=0.1
NUM_EPOCHS=50

if [ -z $1 ] || [ -z $2 ]; then
    exit 0
fi

set -e

if [ ! -z $6 ]; then
    normalized_output=$input;
    filename=$(echo $input|sed -e 's/\.[^./]*$//');
    normalized_input=$input
    input="${filename}_norm.ply";
    if [ ! -f $input ]; then
        echo "INFO: normalizing input";
        python3 samplers/normalize_mesh.py \
            $normalized_input \
            --output $input;
        echo "The input has been normalized to $input";
    fi
fi

echo "INFO: Evaluating:" $input;
echo "INFO: Output:" $output;

if [ ! -f $output/data/samples.pkl ]; then
    echo "INFO: Sampling mesh";
    python3 samplers/sample_mesh.py \
        $input \
        $output/data \
        --voxel_size $VOXLE_SIZE \
        --transformer models/transformer.pth;
fi

if [ ! -f $output/aligned/ckpt_49_latents.npy ]; then 
    echo "INFO: Optimizing aligned latent vectors";
    python3 local_shapes/optimize.py \
        $network/aligned/latest_model.pth \
        $output/data \
        $output/aligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --orient \
        --num_epochs $NUM_EPOCHS \
        --gt_mesh $input \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_49_latents.npy ]; then
    echo "INFO: Optimizing unaligned latent vectors";
    python3 local_shapes/optimize.py \
        $network/unaligned/latest_model.pth \
        $output/data \
        $output/unaligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --num_epochs $NUM_EPOCHS \
        --gt_mesh $input \
        --latent_size $LATENT_SIZE;
fi
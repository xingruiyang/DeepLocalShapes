#!/usr/bin/bash

input=$1;
output=$2;
LATENT_SIZE=64;
CLAMP_DIST=-1;
VOXLE_SIZE=0.1;

if [ -z $1 ] || [ -z $2 ]; then
    exit 0;
fi

set -e

if [ ! -z $3 ]; then
    normalized_output=$input;
    if [[ ! $normalized_output == *.ply ]]; then
        filename=$(echo $input|sed -e 's/\.[^./]*$//');
        normalized_input=$input
        input="${filename}_norm.ply";
        if [ ! -f $normalized_output ]; then
            echo "INFO: normalizing input";
            python3 samplers/normalize_mesh.py \
                $normalized_input \
                --output $input;
            echo "The input has been normalized to $input";
        fi
    fi
fi

echo "INFO: Evaluating:" $input;
echo "INFO: Output:" $output;

if [ ! -f $output/data/samples.pkl ]; then
    echo "INFO: Sampling mesh";
    python3 samplers/sample_mesh.py $input $output/data \
        --voxel_size $VOXLE_SIZE \
        --transformer models/transformer.pth;
fi

if [ ! -f $output/aligned/ckpt_epoch_99_model.pth ]; then
    echo "INFO: Training aligned network";
    python3 local_shapes/trainer.py \
        $output/data \
        $output/aligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --orient \
        --gt_mesh $input \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_epoch_99_model.pth ]; then
    echo "INFO: Training unaligned network";
    python3 local_shapes/trainer.py \
        $output/data \
        $output/unaligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --gt_mesh $input \
        --latent_size $LATENT_SIZE;
fi
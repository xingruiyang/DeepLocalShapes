#!/usr/bin/bash

input=$1;
output=$2;
LATENT_SIZE=$3;
CLAMP_DIST=$4;
VOXLE_SIZE=0.1;
NUM_EPOCHS=100;
BATCH_SIZE=10000;

if [ -z $1 ] || [ -z $2 ]; then
    exit 0;
fi

set -e

if [ ! -z $5 ]; then
    normalized_output=$input;
    if [[ ! $normalized_output == *.ply ]]; then
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
fi

echo "INFO: Evaluating:" $input;
echo "INFO: Output:" $output;

if [ ! -f $output/samples.pkl ]; then
    echo "INFO: Sampling mesh";
    python3 samplers/sample_mesh.py \
        $input \
        $output \
        --voxel_size $VOXLE_SIZE \
        --transformer models/transformer.pth;
fi

if [ ! -f $output/aligned/ckpt_99_model.pth ]; then
    echo "INFO: Training aligned network";
    python3 deep_sdf/trainer.py \
        $output \
        $output/aligned \
        --batch_size $BATCH_SIZE \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 10 \
        --orient \
        --gt_mesh $input \
        --num_epochs $NUM_EPOCHS \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_99_model.pth ]; then
    echo "INFO: Training unaligned network";
    python3 deep_sdf/trainer.py \
        $output \
        $output/unaligned \
        --batch_size $BATCH_SIZE \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 10 \
        --gt_mesh $input \
        --num_epochs $NUM_EPOCHS \
        --latent_size $LATENT_SIZE;
fi
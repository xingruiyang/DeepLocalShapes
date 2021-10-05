#!/usr/bin/bash

input=$1;
output=$2;
LATENT_SIZE=$3;
CLAMP_DIST=$4;
VOXLE_SIZE=0.1;
NUM_EPOCHS=100;
CKPT_FREQ=10;
BATCH_SIZE=10000;

if [ -z $1 ] || [ -z $2 ]; then
    exit 0;
fi

set -e

if [ ! -f $output/aligned/ckpt_99_model.pth ]; then
    echo "INFO: Training aligned network";
    python3 deep_sdf/trainer.py \
        $input \
        $output/aligned \
        --batch_size $BATCH_SIZE \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq $CKPT_FREQ \
        --orient \
        --gt_mesh $input/gt.ply \
        --num_epochs $NUM_EPOCHS \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_99_model.pth ]; then
    echo "INFO: Training unaligned network";
    python3 deep_sdf/trainer.py \
        $input \
        $output/unaligned \
        --batch_size $BATCH_SIZE \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq $CKPT_FREQ \
        --gt_mesh $input/gt.ply \
        --num_epochs $NUM_EPOCHS \
        --latent_size $LATENT_SIZE;
fi
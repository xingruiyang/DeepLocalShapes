#!/usr/bin/bash

input=$1
output=$2
LATENT_SIZE=64
CLAMP_DIST=-1
VOXLE_SIZE=0.1

if [ -z $1 ] || [ -z $2 ]; then
    exit 0
fi

echo "INFO: Evaluating:" $input;
echo "INFO: Output: " $output;

set -e

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
        --output $output/aligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --orient \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_epoch_99_model.pth ]; then
    echo "INFO: Training unaligned network";
    python3 local_shapes/trainer.py \
        $output/data \
        --output $output/unaligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --latent_size $LATENT_SIZE;
fi

for i in {0..99}
do 
    if [ ! -f $output/aligned/ckpt_epoch_${i}_mesh.ply ]; then
        echo "INFO: Reconstructing aligned shapes: ${i}";
        python3 local_shapes/reconstruct.py \
            $output/data \
            $output/aligned/ckpt_epoch_${i}_latents.npy \
            $output/aligned/ckpt_epoch_${i}_model.pth \
            --orient \
            --interp \
            --latent_size $LATENT_SIZE \
            --output $output/aligned/ckpt_epoch_${i}_mesh.ply;
    fi
done

for i in {0..99}
do 
    if [ ! -f $output/unaligned/ckpt_epoch_${i}_mesh.ply ]; then
        echo "INFO: Reconstructing unaligned shapes: ${i}";
        python3 local_shapes/reconstruct.py \
            $output/data \
            $output/unaligned/ckpt_epoch_${i}_latents.npy \
            $output/unaligned/ckpt_epoch_${i}_model.pth \
            --orient \
            --interp \
            --latent_size $LATENT_SIZE \
            --output $output/unaligned/ckpt_epoch_${i}_mesh.ply;
    fi
done

if [ ! -f $output/aligned/chamfer.txt ]; then
    for i in {0..99}
    do 
        echo "INFO: Calculating aligned chamfer distance: ${i}";
        python3 benchmark/cal_chamfer.py \
            $input \
            $output/aligned/ckpt_epoch_${i}_mesh.ply
            >> $output/aligned/chamfer.txt;
    done
fi

if [ ! -f $output/unaligned/chamfer.txt ]; then
    for i in {0..99}
    do 
        echo "INFO: Calculating unaligned chamfer distance: ${i}";
        python3 benchmark/cal_chamfer.py \
            $input \
            $output/unaligned/ckpt_epoch_${i}_mesh.ply
            >> $output/unaligned/chamfer.txt;
    done
fi

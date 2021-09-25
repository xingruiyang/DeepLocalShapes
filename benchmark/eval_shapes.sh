#!/usr/bin/bash

input=$1
output=$2
LATENT_SIZE=64
CLAMP_DIST=-1

if [ -z $1 ] || [ -z $2 ]; then
    exit 0
fi

echo "INFO: Evaluating:" $input;
echo "INFO: Output: " $output;

set -e

if [ ! -f $output/data/samples.pkl ]; then
    echo "INFO: Sampling mesh";
    python3 samplers/sample_mesh.py $input $output/data \
        --voxel_size 0.1 \
        --normalize \
        --transformer models/transformer.pth;
fi

if [ ! -f $output/aligned/ckpt_99_latents.npy ]; then 
    echo "INFO: Optimizing aligned latent vectors";
    python3 local_shapes/optimize.py \
        output/train/aligned/latest_model.pth \
        $output/data \
        $output/aligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --orient \
        --latent_size $LATENT_SIZE;
fi

if [ ! -f $output/unaligned/ckpt_99_latents.npy ]; then
    echo "INFO: Optimizing unaligned latent vectors";
    python3 local_shapes/optimize.py \
        output/train/unaligned/latest_model.pth \
        $output/data \
        $output/unaligned \
        --batch_size 10000 \
        --clamp_dist $CLAMP_DIST \
        --ckpt_freq 1 \
        --latent_size $LATENT_SIZE;
fi

for i in {0..99}
do 
    if [ ! -f $output/aligned/ckpt_${i}_mesh.ply ]; then
        echo "INFO: Reconstructing aligned shapes: ${i}";
        python3 local_shapes/reconstruct.py \
            $output/data \
            $output/aligned/ckpt_${i}_latents.npy \
            output/train/aligned/latest_model.pth \
            --orient \
            --interp \
            --latent_size $LATENT_SIZE \
            --output $output/aligned/ckpt_${i}_mesh.ply;
    fi
done

for i in {0..99}
do 
    if [ ! -f $output/unaligned/ckpt_${i}_mesh.ply ]; then
        echo "INFO: Reconstructing unaligned shapes: ${i}";
        python3 local_shapes/reconstruct.py \
            $output/data \
            $output/unaligned/ckpt_${i}_latents.npy \
            output/train/unaligned/latest_model.pth \
            --interp \
            --latent_size $LATENT_SIZE \
            --output $output/unaligned/ckpt_${i}_mesh.ply;
    fi
done

if [ ! -f $output/aligned/chamfer.txt ]; then
    for i in {0..99}
    do 
        echo "INFO: Calculating aligned chamfer distance: ${i}";
        python3 benchmark/cal_chamfer.py \
            $input \
            $output/aligned/ckpt_${i}_mesh.ply \
            --normalize \
            >> $output/aligned/chamfer.txt;
    done
fi

if [ ! -f $output/unaligned/chamfer.txt ]; then
    for i in {0..99}
    do 
        echo "INFO: Calculating unaligned chamfer distance: ${i}";
        python3 benchmark/cal_chamfer.py \
            $input \
            $output/unaligned/ckpt_${i}_mesh.ply \
            --normalize \
            >> $output/unaligned/chamfer.txt;
    done
fi
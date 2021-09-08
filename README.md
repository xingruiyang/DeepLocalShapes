Deep Local Shapes
=== 

An attempt to replicate the work of [deep local shapes](https://arxiv.org/abs/2003.10983). Since the original authors didn't open-source their code, I decided to write my own. Please note this repo may not represent the quality of the original work, nor does it cover every aspect of the paper.

|depth map|normal map|
|:--:|:--:|
|![img](.github/depth.png)|![img](.github/normal.png)|

## Install

Tested with `CUDA 11.1 update 1` and `Pytorch 1.9`

```
pip install -f requirements.txt
```

## Data Preparation

1. Use `sampler.py` to generate training/evaluation data. 
2. We provide two types of sampler: a `MeshSampler` and a `DepthSampler`.
3. A standalone example can be generated with `python sampler.py input/bun_zipper.ply output/ --voxel_size 0.01`
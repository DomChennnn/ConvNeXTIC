# ConvNeXTIC for VCIP2022

 This repository is the our solution for VCIP2022. The model is modified based on convNeXt.

## Dependencies

### 1. Environment

PyTorch = 1.8

python = 3.8

The docker environment is recommended in [Docker Hub](https://registry.hub.docker.com/layers/pytorch/pytorch/1.8.1-cuda11.1-cudnn8-devel/images/sha256-024af183411f136373a83f9a0e5d1a02fb11acb1b52fdcf4d73601912d0f09b1?context=explore).

### 2. Install dependent packages

```
pip install requirements
```

### 3. how to use the model

```
# train
python train.py --config=./config.yaml --name=nic_cvt # --resume '/workspace/lm/TIC/results/nic_cvt/8/snapshots/best.pt'

# eval
python eval.py --snapshot='./results/tic_/8/snapshots/best.pt' --quality=8

# compress
python codec.py encode --model nic -m mse -q 1 -o compressed.bin test.png

# decompress
python codec.py decode -o out.png compressed.bin
```

## model details

some changes(* for better and x for worse)

1、delete the LayerNorm after or before the downsample_layer and upsample_layer(*)

2、delete the LayerNorm in convnext layer or replace it with gdn

3、use the depth like [2,4,6,2,2,2] like TinyLIC or use the depth like [3,3,9,3,3,3] like swintransform and convnext

4、to be continue

## Citation

```
@article{lu2022high,
  title={High-Efficiency Lossy Image Coding Through Adaptive Neighborhood Information Aggregation},
  author={Lu, Ming and Ma, Zhan},
  journal={arXiv preprint arXiv:2204.11448},
  year={2022}
}
```

## Acknowledgement

The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we thank the authors for their great work.

### 
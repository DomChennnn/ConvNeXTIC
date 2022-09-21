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

## Details

baseline:

| training techniques          |
| ---------------------------- |
| optim: Adam                  |
| data aug:random crop         |
| regularization schemes: None |





| Macro Design                                                 |
| ------------------------------------------------------------ |
| state compute ratio: 2,4,6,2,2,2                             |
| chaneging stem to patchify: use 4x4 non-overlapping convolution |



| Model Detail                                     |
| ------------------------------------------------ |
| depthwise convolution: Yes                       |
| large kernel size: 7x7                           |
| GELU                                             |
| LN                                               |
| delete the LN in downsample and upsample         |
| no drop out                                      |
| convnext layer in encode and decode are the same |



1、delete the LayerNorm after or before the downsample_layer and upsample_layer(*)

2、delete the LayerNorm in convnext layer or replace it with gdn(dong now)

3、use the depth like [2,4,6,2,2,2] like TinyLIC or use the depth like [3,3,9,3,3,3] like swintransform and convnext(dong now)

4、convert dwconv in decode to (dong now)

5、 Adam->AdamW(dong now)

6、smaller kernel size (all 7x7 -> all 5x5 ->7x7 and 5x5)(dong now)

7、use group conv or not 

8、add dropout(dong now)

9、delete the LayerNorm in convnext layer or replace it with BN(dong now)

10、convert mlp ratio from 4 to 2 

11、use normal downsample and upsamle(dong now)

12、the first downsample 4x4 s4 -> 2x2 s2(useful)

8、to be continue

| depth | depth[2,4,6,2,2,2] *baseline | depth[3,3,9,3,3,3] |
| ----- | ---------------------------- | ------------------ |
| q3    |                              |                    |
| q7    |                              |                    |



| kernel size | all 7x7  with dropout | all 5x5 with dropout |
| ----------- | --------------------- | -------------------- |
| q3          |                       |                      |
| q7          |                       |                      |



| act  | GDN  | LN*baseline |
| ---- | ---- | ----------- |
| q3   |      |             |
| q7   |      |             |



| optim | Adam*baseline | AdamW |
| ----- | ------------- | ----- |
| q3    |               |       |
| q7    |               |       |

$$
\begin{aligned}
            \eta_t & = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1
            + \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right),
            & T_{cur} \neq (2k+1)T_{max}; \\
            \eta_{t+1} & = \eta_{t} + \frac{1}{2}(\eta_{max} - \eta_{min})
            \left(1 - \cos\left(\frac{1}{T_{max}}\pi\right)\right),
            & T_{cur} = (2k+1)T_{max}.
        \end{aligned}
$$



| dropout | w    | w/o  |
| ------- | ---- | ---- |
| q3      |      |      |
| q7      |      |      |



| dw in decode | dwconv | dw convtrans |
| ------------ | ------ | ------------ |
| q3           |        |              |
| q7           |        |              |



the performance on this epoch

| epoch                | 50              | 50              | 100             | 100            | 150             | 150             | 200             | 200             |
| -------------------- | --------------- | --------------- | --------------- | -------------- | --------------- | --------------- | --------------- | --------------- |
| Qp                   | 3               | 7               | 3               | 7              | 3               | 7               | 3               | 7               |
| baseline             | 0.3006  27.6162 | 0.9243 30.7690  | 0.3020 28.0753  | 0.9305 31.3303 | 0.3003 28.2833  | 0.9443 31.4792  | 0.3024 28.4361  | 0.9548 31.7102  |
| AdamW                | 0.2971 27.5307  | 0.9101 30.6504  | 0.2948  27.9760 | 0.8942 31.2555 | 0.2941 28.1646  | 0.9057  31.5413 | 0.2947 28.2912  | 0.9149 31.7587  |
| d3393                | 0.3009 27.6640  | 0.9235 30.8117  | 0.2982 28.1132  | 0.9291 31.4262 | 0.3008  28.3752 | 0.9383 31.5835  | 0.3025 28.4938  | 0.9365 31.6994  |
| dwtrans              | 0.3062 27.8559  | 0.9447  31.0872 | 0.3050 28.2050  | 0.9507 31.4511 | 0.3033  28.3577 | 0.9674 31.5568  | 0.3026 28.4055  | 0.9700  31.7013 |
| GDN                  | 0.3037 27.1112  | 0.9484 29.9829  | 0.2950 27.5488  | 0.9139 30.8554 | 0.2937  27.9250 | 0.9137 31.2041  | 0.2943 28.1276  | 0.9178 31.4390  |
| k5(based on dropout) | 0.3040 27.7941  | 0.9363 30.9046  | 0.3014 28.1731  | 0.9425 31.4546 | 0.3028 28.3533  | 0.9512 31.5782  | 0.3032  28.4232 | 0.9552 31.6454  |
| with_dropout         | 0.3046  27.8104 | 0.9298 31.0738  | 0.3027 28.2048  | 0.9359 31.4205 | 0.3038 28.3513  | 0.9529 31.6309  | 0.3044 28.3524  | 0.9539 31.6802  |



the best performance before this epoch

| epoch                | 50              | 50              | 100            | 100            | 150             | 150             | 200             | 200             |
| -------------------- | --------------- | --------------- | -------------- | -------------- | --------------- | --------------- | --------------- | --------------- |
| Qp                   | 3               | 7               | 3              | 7              | 3               | 7               | 3               | 7               |
| baseline             | 0.3006  27.6162 | 0.9247 30.8222  | 0.3020 28.0753 | 0.9271 31.3988 | 0.3008 28.2906  | 0.9456  31.5997 | 0.3024 28.4361  | 0.9578 31.7330  |
| AdamW(maybe)         | 0.2980 27.5829  | 0.9101 30.6504  | 0.2955 27.9648 | 0.8942 31.2555 | 0.2941  28.1587 | 0.9025 31.5043  | 0.2955 28.2936  | 0.9127 31.7412  |
| d3393                | 0.3008 27.6733  | 0.9234 30.8240  | 0.3010 28.1643 | 0.9291 31.4262 | 0.3008  28.3752 | 0.9352 31.6276  | 0.3012 28.4986  | 0.9392 31.7600  |
| dwtrans(maybe)       | 0.3062 27.8559  | 0.9447  31.0872 | 0.3050 28.2050 | 0.9494 31.4361 | 0.3033  28.3577 | 0.9560 31.6347  | 0.3055  28.4401 | 0.9704 31.7289  |
| GDN (worse)          | 0.3031 27.1068  | 0.9484 29.9829  | 0.2950 27.5488 | 0.9139 30.8554 | 0.2936 27.9367  | 0.9117 31.3106  | 0.2943 28.1276  | 0.9178 31.4390  |
| k5(based on dropout) | 0.3040 27.7941  | 0.9363 30.9046  | 0.3014 28.1731 | 0.9434 31.4719 | 0.3028 28.3533  | 0.9532  31.5952 | 0.3032  28.4232 | 0.9566  31.6977 |
| with_dropout(useful) | 0.3046 27.8104  | 0.9298 31.0738  | 0.3027 28.2048 | 0.9358 31.4429 | 0.3038 28.3513  | 0.9529 31.6309  | 0.3035  28.3910 | 0.9569  31.7238 |



## Acknowledgement

The framework is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI/), we thank the authors for their great work.

### 
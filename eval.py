import os
import sys
import argparse
import math
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_msssim import ms_ssim

from models.nic_cvt import NIC
from utils import get_config


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Transformer-based Image Compression Evaluation')
    parser.add_argument('--snapshot', help='snapshot path', type=str, default='/home/domchen/NIC-CfP-main/ckpts/1/best.pt')
    parser.add_argument('--quality', help='quality', type=str, default='1')
    parser.add_argument('--testset', help='testset path', type=str, default='/home/domchen/Kodak')
    # parser.add_argument('--testset', help='testset path', type=str, default='/workspace/lm/data/Tecnick/TESTIMAGES/RGB/RGB_OR_1200x1200')
    args = parser.parse_args(argv)

    #args.config = os.path.join('/workspace/lm/TIC/results/nic_cvt', str(args.quality), 'config.yaml')
    args.config = os.path.join('/home/domchen/NIC-CfP-main','config.yaml')
    return args


def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out):
    size = out['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
              for likelihoods in out['likelihoods'].values()).item()


def pad(x, p=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )


def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def test_kodak(testset_path, model):
    device = next(model.parameters()).device

    psnr_sum = 0.0
    msssim_sum = 0.0
    bit_sum = 0.0
    for i in range(len(os.listdir(testset_path))):
        img = Image.open(testset_path+'/kodim'+str(i+1).zfill(2)+'.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)

        with torch.no_grad():
            out = model.forward(x_pad)

        x_hat = crop(out["x_hat"], (h,w))
        x_hat.clamp_(0, 1)
        rec = transforms.ToPILImage()(x_hat.squeeze().cpu())

        print(f'PSNR: {compute_psnr(x, x_hat):.2f}dB')
        print(f'MS-SSIM: {compute_msssim(x, x_hat):.4f}')
        print(f'Bit-rate: {compute_bpp(out):.3f} bpp')
        psnr_sum += compute_psnr(x, x_hat)
        msssim_sum += compute_msssim(x, x_hat)
        bit_sum += compute_bpp(out)

    print(f'AVG PSNR: {psnr_sum/len(os.listdir(testset_path)):.2f}dB')
    print(f'AVG MS-SSIM: {msssim_sum/len(os.listdir(testset_path)):.4f}')
    print(f'AVG Bit-rate: {bit_sum/len(os.listdir(testset_path)):.3f} bpp')


def test_tecnick(testset_path, model):
    device = next(model.parameters()).device

    psnr_sum = 0.0
    msssim_sum = 0.0
    bit_sum = 0.0
    for i in range(len(os.listdir(testset_path))):
        img = Image.open(testset_path+'/RGB_OR_1200x1200_'+str(i+1).zfill(3)+'.png').convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)

        with torch.no_grad():
            out = model.forward(x_pad)

        x_hat = crop(out["x_hat"], (h,w))
        x_hat.clamp_(0, 1)
        rec = transforms.ToPILImage()(x_hat.squeeze().cpu())

        print(i)
        print(f'PSNR: {compute_psnr(x, x_hat):.2f}dB')
        print(f'MS-SSIM: {compute_msssim(x, x_hat):.4f}')
        print(f'Bit-rate: {compute_bpp(out):.3f} bpp')
        psnr_sum += compute_psnr(x, x_hat)
        msssim_sum += compute_msssim(x, x_hat)
        bit_sum += compute_bpp(out)

    print(f'AVG PSNR: {psnr_sum/len(os.listdir(testset_path)):.2f}dB')
    print(f'AVG MS-SSIM: {msssim_sum/len(os.listdir(testset_path)):.4f}')
    print(f'AVG Bit-rate: {bit_sum/len(os.listdir(testset_path)):.3f} bpp')


def test_clic(testset_path, model):
    device = next(model.parameters()).device

    psnr_sum = 0.0
    msssim_sum = 0.0
    bit_sum = 0.0
    for img_name in (os.listdir(testset_path)):
        img = Image.open(testset_path+'/'+img_name).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
        h, w = x.size(2), x.size(3)
        x_pad = pad(x, p)

        with torch.no_grad():
            out = model.forward(x_pad)

        x_hat = crop(out["x_hat"], (h,w))
        x_hat.clamp_(0, 1)
        rec = transforms.ToPILImage()(x_hat.squeeze().cpu())

        print(img_name)
        print(f'PSNR: {compute_psnr(x, x_hat):.2f}dB')
        print(f'MS-SSIM: {compute_msssim(x, x_hat):.4f}')
        print(f'Bit-rate: {compute_bpp(out):.3f} bpp')
        psnr_sum += compute_psnr(x, x_hat)
        msssim_sum += compute_msssim(x, x_hat)
        bit_sum += compute_bpp(out)

    print(f'AVG PSNR: {psnr_sum/len(os.listdir(testset_path)):.2f}dB')
    print(f'AVG MS-SSIM: {msssim_sum/len(os.listdir(testset_path)):.4f}')
    print(f'AVG Bit-rate: {bit_sum/len(os.listdir(testset_path)):.3f} bpp')


def main(argv):
    args = parse_args(argv)
    config = get_config(args.config)
    # config['embed_num'] = 128
    config['testset'] = args.testset

    print('[config]', args.config)
    msg = f'======================= {args.snapshot} ======================='
    print(msg)
    for k, v in config.items():
        if k in {'lr', 'set_lr', 'p', 'testset'}:
            print(f' *{k}: ', v)
        else:
            print(f'  {k}: ', v)
    print('=' * len(msg))
    print()

    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    snapshot = torch.load(args.snapshot)
    model = NIC(config)
    model.load_state_dict(snapshot['model'])
    model.eval()
    model = model.to(device)
    test_kodak(config['testset'], model)


if __name__ == '__main__':
    main(sys.argv[1:])

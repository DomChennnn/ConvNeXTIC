# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import struct
import sys
import time
import os

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

import compressai

from ckpts import models

from utils import compute_psnr

torch.backends.cudnn.deterministic = True

model_ids = {k: i for i, k in enumerate(models.keys())}

metric_ids = {"mse": 0, "ms-ssim": 1}


def BoolConvert(a):
    b = [False, True]
    return b[int(a)]


def Average(lst):
    return sum(lst) / len(lst)


def inverse_dict(d):
    # We assume dict values are unique...
    assert len(d.keys()) == len(set(d.keys()))
    return {v: k for k, v in d.items()}


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def write_uints(fd, values, fmt=">{:d}I"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 4


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))
    return len(values) * 1


def read_uints(fd, n, fmt=">{:d}I"):
    sz = struct.calcsize("I")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))
    return len(values) * 1


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def get_header(model_name, metric, quality):
    """Format header information:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    metric = metric_ids[metric]
    code = (metric << 4) | (quality - 1 & 0x0F)
    return model_ids[model_name], code


def parse_header(header):
    """Read header information from 2 bytes:
    - 1 byte for model id
    - 4 bits for metric
    - 4 bits for quality param
    """
    model_id, code = header
    quality = (code & 0x0F) + 1
    metric = code >> 4
    return (
        inverse_dict(model_ids)[model_id],
        inverse_dict(metric_ids)[metric],
        quality,
    )


def read_body(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_strings = read_uints(fd, 1)[0]
    for _ in range(n_strings):
        s = read_bytes(fd, read_uints(fd, 1)[0])
        lstrings.append([s])

    return lstrings, shape


def write_body(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings)))
    for s in out_strings:
        bytes_cnt += write_uints(fd, (len(s[0]),))
        bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def write_body_VR(fd, shape, out_strings):
    bytes_cnt = 0
    bytes_cnt = write_uints(fd, (shape[0], shape[1], len(out_strings), len(out_strings[0])))
    for out_string in out_strings:
        for s in out_string:
            bytes_cnt += write_uints(fd, (len(s[0]),))
            bytes_cnt += write_bytes(fd, s[0])
    return bytes_cnt

def read_body_VR(fd):
    lstrings = []
    shape = read_uints(fd, 2)
    n_list = read_uints(fd, 1)[0]
    n_strings = read_uints(fd, 1)[0]
    for num in range(n_list):
        temp_string = []
        for _ in range(n_strings):
            s = read_bytes(fd, read_uints(fd, 1)[0])
            temp_string.append([s])
        lstrings.append(temp_string)

    return lstrings, shape

def pad_VR(x, p_h=2 ** 6, p_w=2 ** 6):
    h, w = x.size(2), x.size(3)
    H = (h + p_h - 1) // p_h * p_h
    W = (w + p_w - 1) // p_w * p_w
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


# def _encode(image, model, metric, quality, coder, output, lamb):
#     compressai.set_entropy_coder(coder)
#     enc_start = time.time()
#
#     img = load_image(image)
#     start = time.time()
#     # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = models[model](quality=quality, metric=metric, pretrained=True).to(device).eval()
#     load_time = time.time() - start
#
#     x = img2torch(img)
#     h, w = x.size(2), x.size(3)
#     p = 256  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
#     x = pad(x, p)
#
#     x = x.to(device)
#     lamb = np.array([lamb],np.float32)
#     lamb = torch.Tensor(lamb).cuda()
#     # print(lamb)
#     # net = net.cuda()
#     with torch.no_grad():
#         out = net.compress(x, lamb)
#
#     shape = out["shape"]
#     header = get_header(model, metric, quality)
#
#     with Path(output).open("wb") as f:
#         write_uchars(f, header)
#         # write original image size
#         write_uints(f, (h, w))
#         # write shape and number of encoded latents
#         write_body(f, shape, out["strings"])
#
#     enc_time = time.time() - enc_start
#     size = filesize(output)
#     bpp = float(size) * 8 / (img.size[0] * img.size[1])
#     print(
#         f"{bpp:.3f} bpp |"
#         f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
#     )


def _encode(image, model, metric, quality, coder, output, lamb, p_h, p_w):
    compressai.set_entropy_coder(coder)
    enc_start = time.time()

    img = load_image(image)
    start = time.time()
    # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models[model](quality=quality, metric=metric, pretrained=True).to(device).eval()
    load_time = time.time() - start

    x = img2torch(img)
    h, w = x.size(2), x.size(3)
    # p = 1280  # maximum 6 strides of 2, and window size 4 for the smallest latent fmap: 4*2^6=256
    x = pad_VR(x, p_h, p_w)

    x = x.to(device)
    # print(lamb)
    # net = net.cuda()

    unfold = torch.nn.Unfold(kernel_size=(p_h,p_w),stride=(p_h, p_w))
    # print(x.shape)
    x = unfold(x)
    x = x.permute(0, 2, 1)
    x = x.view(-1, 3, p_h, p_w)
    Patch_Nums = x.shape[0]
    # print(x.shape)
    header = get_header(model, metric, quality)
    lamb_flag = 0 if lamb < 0 else 1

    with Path(output).open("wb") as f:
        write_uchars(f, header)
        # write original image size
        write_uints(f, (h, w, p_h, p_w, lamb_flag, lamb if lamb_flag == 1 else -lamb))
        # write shape and number of encoded latents

    lamb = np.array([lamb/10000],np.float32)
    lamb = torch.Tensor(lamb).cuda()
    print(lamb)
    outstrings = []
    for i in range(Patch_Nums):
        with torch.no_grad():
            out = net.compress(x[[i],:,:,:], lamb)

        outstrings.append(out["strings"])
        shape = out["shape"]


    with Path(output).open("ab") as f:
        write_body_VR(f, shape, outstrings)

    enc_time = time.time() - enc_start
    size = filesize(output)
    bpp = float(size) * 8 / (img.size[0] * img.size[1])
    print(
        f"{bpp:.3f} bpp |"
        f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
    )


def _decode(inputpath, coder, show, output=None):
    compressai.set_entropy_coder(coder)
    with Path(inputpath).open("rb") as f:
        model, metric, quality = parse_header(read_uchars(f, 2))
        original_size = read_uints(f, 2)
        padding_size = read_uints(f, 2)
        lamb_flag = read_uints(f, 1)[0]
        lamb = read_uints(f, 1)[0]
        strings, shape = read_body_VR(f)
    h, w = original_size[0], original_size[1]
    p_h, p_w = padding_size[0], padding_size[1]
    h_pad = int(np.ceil(h/p_h)*p_h)
    w_pad = int(np.ceil(w/p_w)*p_w)
    lamb = lamb if lamb_flag == 1 else -lamb
    print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")

    # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = models[model](quality=quality, metric=metric, pretrained=True).to(device).eval()

    # net = net.cuda()
    lamb = np.array([lamb/10000],np.float32)
    lamb = torch.Tensor(lamb).cuda()
    print(lamb)
    rec = torch.Tensor(np.zeros((len(strings), 3, p_h, p_w))).cuda()
    torch.cuda.synchronize()
    start = time.time()
    for decode_idx in range(len(strings)):
        with torch.no_grad():
            out = net.decompress(strings[decode_idx], shape, lamb)
            rec[[decode_idx],:,:,:] = out["x_hat"]
    torch.cuda.synchronize()
    end = time.time()
    rec = rec.permute(1,2,3,0)
    rec = rec.view(1, 3*p_w*p_h, -1)
    fold = torch.nn.Fold(output_size=(h_pad, w_pad) ,kernel_size=(p_h, p_w), stride=(p_h, p_w))
    # print(x.shape)
    rec = fold(rec)

    x_hat = crop(rec, original_size)
    img = torch2img(x_hat)

    dec_time = end - start
    print(f"Decoded in {dec_time:.2f}s")

    if show:
        show_image(img)
    if output is not None:
        img.save(output)



# def _encode(image, model, metric, quality, coder, output, lamb):
#     compressai.set_entropy_coder(coder)
#     enc_start = time.time()
#
#     img = load_image(image)
#     start = time.time()
#     # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = models[model](quality=quality, metric=metric, pretrained=True).to(device).eval()
#     load_time = time.time() - start
#
#     x = img2torch(img)
#     h, w = x.size(2), x.size(3)
#     p_w = 2 ** np.ceil(np.log2(np.ceil(w/2)))
#     p_h = 2 ** np.ceil(np.log2(np.ceil(h/2)))
#
#     p_w = p_w if p_w < 1536 else 1536
#     p_h = p_h if p_h < 1536 else 1536
#
#     x = pad_VR(x, p_w, p_h)
#
#     x = x.to(device)
#     lamb = np.array([lamb],np.float32)
#     lamb = torch.Tensor(lamb).cuda()
#     # print(lamb)
#     # net = net.cuda()
#
#
#     # print(x.shape)
#     header = get_header(model, metric, quality)
#     with Path(output).open("wb") as f:
#         write_uchars(f, header)
#         # write original image size
#         write_uints(f, (h, w))
#         # write shape and number of encoded latents
#
#     outstrings = []
#     for i in range(Patch_Nums):
#         with torch.no_grad():
#             out = net.compress(x[[i],:,:,:], lamb)
#
#         outstrings.append(out["strings"])
#         shape = out["shape"]
#
#
#     with Path(output).open("ab") as f:
#         write_body_VR(f, shape, outstrings)
#
#     enc_time = time.time() - enc_start
#     size = filesize(output)
#     bpp = float(size) * 8 / (img.size[0] * img.size[1])
#     print(
#         f"{bpp:.3f} bpp |"
#         f" Encoded in {enc_time:.2f}s (model loading: {load_time:.2f}s)"
#     )
#
#
# def _decode(inputpath, coder, show, lamb, output=None):
#     compressai.set_entropy_coder(coder)
#     p = 1024
#     with Path(inputpath).open("rb") as f:
#         model, metric, quality = parse_header(read_uchars(f, 2))
#         original_size = read_uints(f, 2)
#         strings, shape = read_body_VR(f)
#     h, w = original_size[0], original_size[1]
#     h_pad = int(np.ceil(h/p)*p)
#     w_pad = int(np.ceil(w/p)*p)
#     print(f"Model: {model:s}, metric: {metric:s}, quality: {quality:d}")
#
#     # net = models[model](quality=quality, metric=metric, pretrained=True).eval()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = models[model](quality=quality, metric=metric, pretrained=True).to(device).eval()
#     torch.cuda.synchronize()
#     start = time.time()
#     # net = net.cuda()
#     lamb = np.array([lamb],np.float32)
#     lamb = torch.Tensor(lamb).cuda()
#     rec = torch.Tensor(np.zeros((len(strings), 3, p, p))).cuda()
#     for decode_idx in range(len(strings)):
#         with torch.no_grad():
#             out = net.decompress(strings[decode_idx], shape, lamb)
#             rec[[decode_idx],:,:,:] = out["x_hat"]
#     rec = rec.permute(1,2,3,0)
#     rec = rec.view(1, 3*p*p, -1)
#     fold = torch.nn.Fold(output_size=(h_pad, w_pad) ,kernel_size=(p, p), stride=p)
#     # print(x.shape)
#     rec = fold(rec)
#
#     x_hat = crop(rec, original_size)
#     img = torch2img(x_hat)
#     torch.cuda.synchronize()
#     end = time.time()
#     dec_time = end - start
#     print(f"Decoded in {dec_time:.2f}s")
#
#     if show:
#         show_image(img)
#     if output is not None:
#         img.save(output)


def show_image(img: Image.Image):
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.title.set_text("Decoded image")
    ax.imshow(img)
    fig.tight_layout()
    plt.show()


def encode(argv):
    parser = argparse.ArgumentParser(description="Encode image to bit-stream")
    parser.add_argument("--image", type=str,default='/workspace/sharedata/VCIP2022/val/20.png')#'/workspace/Kodak/kodim05.png''/workspace/sharedata/VCIP2022/test/Animal/aries-wild-free-running-wildlife-park-158025_1.png'
    parser.add_argument(
        "--model",
        choices=models.keys(),
        default=list(models.keys())[0],
        help="NN model to use (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--metric",
        choices=metric_ids.keys(),
        default="mse",
        help="metric trained against (default: %(default)s",
    )
    parser.add_argument(
        "-q",
        "--quality",
        choices=list(range(1, 9)),
        type=int,
        default=6,
        help="Quality setting (default: %(default)s)",
    )
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("-o", "--output", help="Output path",default='out.bin')
    parser.add_argument("--lamb", help="adjust fact", default=2140) # lambda*10000//1  0-10000
    parser.add_argument("--patch_size_h", help="up to down patch", default=1344)#1280 should be 64N
    parser.add_argument("--patch_size_w", help="left to right patch", default=1472)#1280 should be 64N
    args = parser.parse_args(argv)

    if not args.output:
        args.output = Path(Path(args.image).resolve().name).with_suffix(".bin")

    _encode(args.image, args.model, args.metric, args.quality, args.coder, args.output, args.lamb, args.patch_size_h, args.patch_size_w)


def decode(argv):
    parser = argparse.ArgumentParser(description="Decode bit-stream to imager")
    parser.add_argument("--input", type=str,  default='out.bin')
    parser.add_argument(
        "-c",
        "--coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="Entropy coder (default: %(default)s)",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("-o", "--output", help="Output path",default='out.png')
    # parser.add_argument("--lamb", help="adjust fact", default=0)
    args = parser.parse_args(argv)
    _decode(args.input, args.coder, args.show, args.output)


def parse_args(argv):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--command", choices=["encode", "decode"],default="decode")
    # parser.add_argument("--image", type=str, default='/workspace/sharedata/VCIP2022/val/1.png')
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv[1:2])
    argv = argv[2:]
    torch.set_num_threads(1)  # just to be sure
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    # if args.command == "encode":
    #     encode(argv)
    # elif args.command == "decode":
    #     decode(argv)

    encode(argv)

    decode(argv)

    from PIL import Image
    path_in = '/workspace/sharedata/VCIP2022/val/20.png'
    path_rec = 'out.png'
    img1 = Image.open(path_in)
    img2 = Image.open(path_rec)
    img1 = np.array(img1)
    img2 = np.array(img2)
    print(compute_psnr(img1,img2))

if __name__ == "__main__":
    main(sys.argv)
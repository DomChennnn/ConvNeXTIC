import os
import imageio
import subprocess
from skimage import img_as_ubyte, img_as_float


class BPGEncoder:
    def __init__(self, qp='29', fmt='420', cp='ycbcr', bit_depth='8', lossless=False, encoder='x265', level='8'):
        '''Main options'''
        self.filename = ''
        self.output = ''
        self.qp = str(qp)  # quantization level [0-51]
        self.cfmt = str(fmt)  # image color format [420, 422, 444]
        self.color_space = cp  # color space [ycbcr, rgb, ycgco, ycbcr_bt709,ycbcr_bt2020]
        self.bit_depth = str(bit_depth)  # image bitdepth [8- 12]
        self.lossless = lossless  # enable lossless entropy coding
        self.encoder = encoder  # enable hevc encoder [x265]
        self.level = str(level)  # compression level [1(fast) - 9(slow)]
        self.bytes = 0

    def encode(self, frame=None, filename=None, output='out.bpg', out=False):
        if filename == None:
            temp_path = os.getcwd() + '/temp/temp.jpg'
            imageio.imsave(temp_path, img_as_ubyte(frame))
            self.filename = temp_path
        else:
            self.filename = filename  # string name to png /jpg image file

        self.output = output + '.bpg'
        if self.lossless:
            cmd = ['bpgenc', self.filename, '-o', self.output, '-q', self.qp, 'f', self.cfmt, '-c', self.color_space,
                   '-b', self.bit_depth, '-e', self.encoder, '-m', self.level, '-lossless']
        else:
            cmd = ['bpgenc', self.filename, '-o', self.output, '-q', self.qp, 'f', self.cfmt, '-c', self.color_space,
                   '-b', self.bit_depth, '-e', self.encoder, '-m', self.level]
        subprocess.call(cmd)
        self.bytes += os.path.getsize(self.output)
        if filename == None:
            os.remove(temp_path)
        if out:
            # temp_dir = os.getcwd()+'/temp.png'
            cmd_1 = ['bpgdec', self.output, '-o', output + '_d.jpg', '-b', self.bit_depth]
            subprocess.call(cmd_1)
            frame = img_as_float(imageio.imread(output + '_d.jpg'))
            # os.remove(temp_dir)
            return frame, self.bytes
        else:
            return self.bytes


class BPGDecoder:
    def __init__(self, bit_depth=8, info=False):
        self.input = ''
        self.bit_depth = str(bit_depth)
        self.info = info
        self.output = ''

    def decode(self, encoded_file, output_file=None):
        self.input = encoded_file
        if output_file != None:
            self.output = output_file
        else:
            self.output = os.getcwd() + '/temp/temp.jpg'
        if self.info:
            cmd = ['bpgdec', self.input, '-o', self.output, '-i', '-b', self.bit_depth]
        else:
            cmd = ['bpgdec', self.input, '-o', self.output, '-b', self.bit_depth]
        subprocess.call(cmd)
        frame = img_as_float(imageio.imread(self.output))
        if output_file == None:
            os.remove(self.output)
        return frame


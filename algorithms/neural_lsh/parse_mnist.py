import struct
import numpy as np
from types import SimpleNamespace


def parse(filename):
    r = _no_numpy(filename)
    return _to_numpy(r)


def _no_numpy(filename):
    with open(filename, 'rb') as f:

        r = SimpleNamespace()

        the_4_unsigned = struct.unpack('>IIII', f.read(16))
        magic_number, r.number_of_images, r.rows, r.cols = the_4_unsigned

        if magic_number != 2051:
            raise Exception(f"'{filename}' is not MNIST, the magic number is 2051 but {magic_number} was read")

        r.pixel_brightnesses = f.read(r.number_of_images * r.rows * r.cols)

        return r


def _to_numpy(r):
    as_numpy = np.frombuffer(r.pixel_brightnesses, dtype=np.uint8)
    return as_numpy.copy().reshape(r.number_of_images, r.rows * r.cols).astype(np.float32, copy=False)


_f = './input_data/t10k-images.idx3-ubyte'  # helping interactive use


if __name__ == '__main__':
    r = _no_numpy(_f)
    images = _to_numpy(r)
    while True:
        i = None
        try:
            i = int(input(f"show image (0-{len(images) - 1}): "))
        except Exception:
            print('(exiting)')
            break
        image = images[i]
        to_print = ''
        m = 0
        for pixel in images[i]:
            background = int(int(pixel) * 23 / 255)  # from 0-255 to 0-23, int(pixel) is necessary
            chars = '. '
            to_print += f"\x1b[48;5;{232 + background}m{chars}"  # for ANSI terminal
            m += 1
            if m == r.cols:
                to_print += '\x1b[0m\n'  # new line, resetting the background color
                m = 0
        to_print += '\x1b[0m'
        print(to_print)

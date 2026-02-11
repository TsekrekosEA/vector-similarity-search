import numpy as np
from types import SimpleNamespace
import struct

def parse(filename):
    with open(filename, "rb") as f:
        blob = f.read()        

    offset = 0
    images = []

    r = SimpleNamespace()

    while offset < len(blob):
        n_pix = struct.unpack_from("<I", blob, offset)[0]
        offset += 4

        size = n_pix * 4
        img = np.frombuffer(blob, dtype="<f4", count=n_pix, offset=offset).copy()
        offset += size

        images.append(img)

    excess = len(blob) - offset
    if excess != 0:
        raise Exception(f"'{filename}', parsed as a SIFT file, has {excess} excess bytes")

    return np.stack(images)


_f = './input_data/sift_learn.fvecs'  # helping interactive use


if __name__ == "__main__":
    images = parse(_f)
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
            background = int(pixel * 23 / 255)  # from 0-255 to 0-23
            chars = '. '
            to_print += f"\x1b[48;5;{232 + background}m{chars}"  # for ANSI terminal
            m += 1
            if m == 32:
                to_print += '\x1b[0m\n'  # new line, resetting the background color
                m = 0
        to_print += '\x1b[0m'
        print(to_print)

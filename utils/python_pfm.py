import re
import numpy as np
import sys


def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(br'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    # greyscale
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n'.encode())
    file.write(('%d %d\n' % (image.shape[1], image.shape[0])).encode())

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode())

    image.tofile(file)


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    import cv2

    im_a = cv2.imread(
        '/home/gu/workspace/Gu/EANet/EANet_0.3/dataloader/myDataset/Adirondack_l.png')
    im_b = cv2.imread(
        '/home/gu/workspace/Gu/EANet/EANet_0.3/dataloader/myDataset/Jadeplant_l.png')

    im_left = np.empty((512, 768, 3), dtype=np.uint8)
    im_right = np.empty((512, 768, 3), dtype=np.uint8)
    disp_left = np.empty((512, 768), dtype=np.float32)

    disp1 = 30
    disp2 = 100

    assert disp1 != 0 and disp2 != 0

    im_left[:256] = im_a[:256, -768:]
    im_left[256:] = im_b[256:512, -768:]
    im_right[:256] = im_a[:256, -768-disp1:-disp1]
    im_right[256:] = im_b[256:512, -768-disp2:-disp2]
    disp_left[:256] = disp1
    disp_left[256:] = disp2

    plt.subplot(1, 3, 1), plt.imshow(im_left)
    plt.subplot(1, 3, 2), plt.imshow(im_right)
    plt.subplot(1, 3, 3), plt.imshow(disp_left, cmap='gray', vmin=0, vmax=768)
    plt.show()

    cv2.imwrite(
        '/home/gu/workspace/Gu/EANet/EANet_0.3/dataloader/myDataset/im0.png', im_left)
    cv2.imwrite(
        '/home/gu/workspace/Gu/EANet/EANet_0.3/dataloader/myDataset/im1.png', im_right)
    writePFM(
        '/home/gu/workspace/Gu/EANet/EANet_0.3/dataloader/myDataset/disp0GT.pfm', disp_left)

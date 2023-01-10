import numpy as np
import timeit
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern


def lbp(input_img: np.ndarray, points: int, radius: float) -> np.ndarray:
    height, width = input_img.shape
    output = np.zeros((height, width))
    int_radius = np.ceil(radius).astype(int)
    padded_input = np.pad(input_img, ((int_radius, int_radius), (int_radius, int_radius)), 'constant',
                          constant_values=(0, 0))
    for cy in range(height):  # range(int_radius, height - int_radius)
        for cx in range(width):  # range(int_radius, width - int_radius)
            p = np.arange(points)
            px = cx - radius * np.sin(2 * np.pi * p / points)
            py = cy + radius * np.cos(2 * np.pi * p / points)
            # TODO: vettorizza interpolazione (se possibile)
            # p_val = bilinear_interpolation(padded_input, px + int_radius, py + int_radius)
            p_val = np.zeros(points)
            for i in range(points):
                p_val[i] = bilinear_interpolation(padded_input, px[i] + int_radius, py[i] + int_radius)
            # p_val = [bilinear_interpolation(padded_input, x + int_radius, y + int_radius) for x, y in zip(px, py)]
            signed_val = p_val >= input_img[cy, cx]
            lbp_val = np.sum(signed_val * 2 ** p)
            output[cy, cx] = lbp_val
    output = (np.rint(output)).astype(np.uint8)
    return output


def bilinear_interpolation(input_img: np.ndarray, px, py) -> float:
    x1 = np.floor(px).astype(int)
    x2 = np.ceil(px).astype(int)
    y1 = np.floor(py).astype(int)
    y2 = np.ceil(py).astype(int)

    q11 = input_img[y1, x1]
    q12 = input_img[y2, x1]
    q22 = input_img[y2, x2]
    q21 = input_img[y1, x2]

    if np.abs(x1 - x2) < 1e-6 and not np.abs(y1 - y2) < 1e-6:
        return q11 * ((y2 - py) / (y2 - y1)) + q12 * ((py - y1) / (y2 - y1))
    elif not np.abs(x1 - x2) < 1e-6 and np.abs(y1 - y2) < 1e-6:
        return q11 * ((x2 - px) / (x2 - x1)) + q22 * ((px - x1) / (x2 - x1))
    elif np.abs(x1 - x2) < 1e-6 and np.abs(y1 - y2) < 1e-6:
        return input_img[y1, x1]
    else:
        return (1 / (x2 - x1) * (y2 - y1)) * (
                q11 * (x2 - px) * (y2 - py)
                + q21 * (px - x1) * (y2 - py)
                + q12 * (x2 - px) * (py - y1)
                + q22 * (px - x1) * (py - y1)
        )


if __name__ == '__main__':
    img = Image.open(r"./test.jpg").convert("L")
    in_img = np.asarray(img)

    pts = 8
    rd = 1

    start_time = timeit.default_timer()
    output_img = lbp(in_img, pts, rd)
    end_time = timeit.default_timer()
    print("Time (s): ", end_time - start_time)

    plt.hist(output_img.flatten(), bins=2**8)
    plt.savefig('hist.png')

    plt.clf()

    correct_lbp = local_binary_pattern(in_img, pts, rd, method="default")

    correct_lbp = (np.rint(correct_lbp)).astype(np.uint8)
    corr_img = Image.fromarray(correct_lbp)
    corr_img.save('./res-cor.jpg', 'jpeg')

    plt.hist(correct_lbp.ravel(), bins=2**8)
    plt.savefig('hist-cor.png')

    out_img = Image.fromarray(output_img)
    out_img.save('./res.jpg', 'jpeg')

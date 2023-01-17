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
    for cy in range(height):
        for cx in range(width):
            lbp_val = 0
            p_vals = []
            for p in range(points):
                px = cx - radius * np.sin(2 * np.pi * p / points)
                py = cy + radius * np.cos(2 * np.pi * p / points)
                p_val = bilinear_interpolation(padded_input, px + int_radius, py + int_radius)
                p_vals.append(p_val)
                if p_val >= input_img[cy, cx]:
                    lbp_val += 1
            s = np.heaviside(np.subtract(p_vals, input_img[cy, cx]), 1)
            u = np.sum(np.abs(s[1:] - s[:-1]))
            if u > 2:
                lbp_val = points + 1
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

    if x1 == x2 and y1 != y2:
        return q11 * ((y2 - py) / (y2 - y1)) + q12 * ((py - y1) / (y2 - y1))
    elif x1 != x2 and y1 == y2:
        return q11 * ((x2 - px) / (x2 - x1)) + q22 * ((px - x1) / (x2 - x1))
    elif x1 == x2 and y1 == y2:
        return input_img[y1, x1]
    else:
        return (1 / (x2 - x1) * (y2 - y1)) * (
                q11 * (x2 - px) * (y2 - py)
                + q21 * (px - x1) * (y2 - py)
                + q12 * (x2 - px) * (py - y1)
                + q22 * (px - x1) * (py - y1)
        )


if __name__ == '__main__':
    img = Image.open(r"./input/test.jpg").convert("L")
    in_img = np.asarray(img)

    pts = 8
    rd = 1

    start_time = timeit.default_timer()
    output_img = lbp(in_img, pts, rd)
    end_time = timeit.default_timer()
    print("Time (s): ", end_time - start_time)

    plt.hist(output_img.flatten(), bins=pts + 2)
    plt.savefig('output/hist.png')

    # Scikit-image comparison
    # plt.clf()
    #
    # correct_lbp = local_binary_pattern(in_img, pts, rd, method="uniform")
    #
    # correct_lbp = (np.rint(correct_lbp)).astype(np.uint8)
    # corr_img = Image.fromarray(correct_lbp)
    # corr_img.save('./output/res-cor.jpg', 'jpeg')
    #
    # plt.hist(correct_lbp.ravel(), bins=pts + 2)
    # plt.savefig('output/hist-cor.png')

    out_img = Image.fromarray(output_img)
    out_img.save('./output/res.jpg', 'jpeg')

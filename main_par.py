import numpy as np
import timeit
from PIL import Image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, dump, load
from tempfile import mkdtemp
import os


def lbp(input_img: np.ndarray, points: int, radius: float, proc: int) -> np.ndarray:
    height, width = input_img.shape

    savedir = mkdtemp()
    input_path = os.path.join(savedir, 'input.joblib')
    dump(input_img, input_path, compress=True)

    slice_height = np.ceil(height / proc).astype(int)
    output_slices = Parallel(n_jobs=proc)(delayed(sub_image_lbp)
                                          (input_path, points, radius, y, y + slice_height, 0, width)
                                          for y in range(0, height, slice_height))

    output = output_slices[0]
    for i in range(1, len(output_slices)):
        if output_slices[i] is not None:
            output = np.vstack((output, output_slices[i]))

    os.remove(input_path)
    output = (np.rint(output)).astype(np.uint8)
    return output


def sub_image_lbp(input_path: str, points: int, radius: float, start_y: int, end_y: int, start_x: int,
                  end_x: int) -> np.ndarray | None:
    input_img = load(input_path)
    height, width = input_img.shape

    start_x = start_x if start_x > -1 else 0
    start_y = start_y if start_y > -1 else 0
    end_x = end_x if end_x <= width else width
    end_y = end_y if end_y <= height else height

    if start_x >= width or start_y >= height or end_x <= 0 or end_y <= 0:
        return None

    int_radius = np.ceil(radius).astype(int)
    padded_input = np.pad(input_img, ((int_radius, int_radius), (int_radius, int_radius)), 'constant',
                          constant_values=(0, 0))

    slice_height = end_y - start_y
    slice_width = end_x - start_x
    output = np.zeros((slice_height, slice_width))

    for cy in range(slice_height):
        for cx in range(slice_width):
            lbp_val = 0
            p_vals = []
            for p in range(points):
                px = cx + start_x - radius * np.sin(2 * np.pi * p / points)
                py = cy + start_y + radius * np.cos(2 * np.pi * p / points)
                p_val = bilinear_interpolation(padded_input, px + int_radius, py + int_radius)
                p_vals.append(p_val)
                if p_val >= input_img[cy + start_y, cx + start_x]:
                    lbp_val += 1
            s = np.heaviside(np.subtract(p_vals, input_img[cy + start_y, cx + start_x]), 1)
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

    output_img = None

    n_rep = 1
    times = []
    for p in range(5):
        processes = 2 ** p
        start_time = timeit.default_timer()
        for n in range(n_rep):
            output_img = lbp(in_img, pts, rd, processes)
        end_time = timeit.default_timer()
        time = (end_time - start_time) / n_rep
        times.append((processes, time))
        print("Time (s):", time, ", Processes:", processes)
    print("Process-Times:", times)

    plt.hist(output_img.flatten(), bins=pts + 2)
    plt.savefig('output/hist.png')

    out_img = Image.fromarray(output_img)
    out_img.save('./output/res.jpg', 'jpeg')

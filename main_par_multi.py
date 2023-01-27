import os
from os import listdir
from tempfile import mkdtemp

import numpy as np
import timeit
from PIL import Image
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, load, dump


def lbp_multi(input_list: list[np.ndarray], points: int, radius: float, proc: int) -> list[np.ndarray]:
    # return Parallel(n_jobs=proc)(delayed(lbp)(input_img, points, radius) for input_img in input_list)
    savedir = mkdtemp()
    input_paths = []
    for i in range(len(input_list)):
        input_path = os.path.join(savedir, f'input{i}.joblib')
        dump(input_list[i], input_path, compress=True)
        input_paths.append(input_path)

    outputs = Parallel(n_jobs=proc)(delayed(lbp_single_load)(input_path, points, radius)
                                    for input_path in input_paths)

    for path in input_paths:
        os.remove(path)
    return outputs


def lbp_single_load(input_path: str, points: int, radius: float) -> np.ndarray:
    input_img = load(input_path)
    return lbp(input_img, points, radius)


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
    input_dir = './input_dataset/'
    output_dir = './output_dataset/'
    pts = 8
    rd = 1
    max_img = 20

    input_list = []
    for filename, _ in zip(listdir(input_dir), range(max_img)):
        img = Image.open(input_dir + filename).convert("L")
        in_img = np.asarray(img)
        input_list.append(in_img)

    output_list = []

    n_rep = 1
    times = []
    for p in range(5):
        processes = 2 ** p
        start_time = timeit.default_timer()
        for n in range(n_rep):
            output_list = lbp_multi(input_list, pts, rd, processes)
        end_time = timeit.default_timer()
        time = (end_time - start_time) / n_rep
        times.append((processes, time))
        print("Time (s):", time, ", Processes:", processes)
    print("Process-Times:", times)

    if len(output_list) > 0:
        for filename in listdir(output_dir):
            os.remove(output_dir + filename)

    for i in range(len(output_list)):
        output_img = output_list[i]
        plt.hist(output_img.flatten(), bins=pts + 2)
        plt.savefig(output_dir + f'hist{i}.png')
        plt.clf()

        out_img = Image.fromarray(output_img)
        out_img.save(output_dir + f'res{i}.jpg', 'jpeg')

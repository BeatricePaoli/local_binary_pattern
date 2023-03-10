import numpy as np
import matplotlib.pyplot as plt


def plot_latex_table(points: np.ndarray) -> None:
    speedup = np.copy(points)
    speedup[:, 1] = points[0, 1] / points[:, 1]
    for p, s in zip(points, speedup):
        print(f'{p[0]:.0f} & {p[1]:.3f} & {s[1]:.3f} \\\\')
    print()


if __name__ == '__main__':
    points1 = np.array([(1, 48.62989480001852), (2, 32.759138499968685), (4, 21.53144459996838), (8, 19.032108999963384), (16, 19.302141599997412)])
    plot_latex_table(points1)
    points1[:, 1] = points1[0, 1] / points1[:, 1]

    points2 = np.array([(1, 97.45241339999484), (2, 63.37841820000904), (4, 43.865591900015716), (8, 37.927015500026755), (16, 38.614296700048726)])
    plot_latex_table(points2)
    points2[:, 1] = points2[0, 1] / points2[:, 1]

    points3 = np.array([(1, 130.88258390000556), (2, 82.11559509998187), (4, 61.13243179995334), (8, 57.4532923999941), (16, 58.26480730000185)])
    plot_latex_table(points3)
    points3[:, 1] = points3[0, 1] / points3[:, 1]

    points4 = np.array([(1, 169.36517579999054), (2, 96.62600739998743), (4, 76.86508229997708), (8, 76.84517650003545), (16, 76.81608899997082)])
    plot_latex_table(points4)
    points4[:, 1] = points4[0, 1] / points4[:, 1]

    points5 = np.array([(1, 5.217770699993707), (2, 3.2199761000229046), (4, 2.2447581999585964), (8, 2.3993399999453686), (16, 2.568057900003623)])
    plot_latex_table(points5)
    points5[:, 1] = points5[0, 1] / points5[:, 1]

    points6 = np.array([(1, 48.62989480001852), (2, 32.759138499968685), (4, 21.53144459996838), (8, 19.032108999963384), (16, 19.302141599997412)])
    plot_latex_table(points6)
    points6[:, 1] = points6[0, 1] / points6[:, 1]

    points7 = np.array([(1, 203.60042450000765), (2, 161.33932620001724), (4, 83.3116122999927), (8, 72.8485697999713), (16, 74.91348990000552)])
    plot_latex_table(points7)
    points7[:, 1] = points7[0, 1] / points7[:, 1]

    points8 = np.array([(1, 272.12910820002435), (2, 213.77167990000453), (4, 122.107477499987), (8, 120.78227149997838), (16, 123.20197950000875)])
    plot_latex_table(points8)
    points8[:, 1] = points8[0, 1] / points8[:, 1]

    points9 = np.array([(1, 189.70019509998383), (2, 126.12823549995665), (4, 89.95179550000466), (8, 85.12617699999828), (16, 77.20998139999574)])
    plot_latex_table(points9)
    points9[:, 1] = points9[0, 1] / points9[:, 1]

    points10 = np.array([(1, 339.4715341000119), (2, 226.98205160000362), (4, 153.17267430003267), (8, 152.7001507000532), (16, 152.56952220003586)])
    plot_latex_table(points10)
    points10[:, 1] = points10[0, 1] / points10[:, 1]

    points11 = np.array([(1, 522.8232376000378), (2, 326.59442639991175), (4, 236.41241500002798), (8, 232.3571268999949), (16, 227.53912790003233)])
    plot_latex_table(points11)
    points11[:, 1] = points11[0, 1] / points11[:, 1]

    points12 = np.array([(1, 693.2188653999474), (2, 435.3789729999844), (4, 306.2648754999973), (8, 300.66049579996616), (16, 306.603324500029)])
    plot_latex_table(points12)
    points12[:, 1] = points12[0, 1] / points12[:, 1]

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    # image test.jpg, r=1
    plt.title('Speedup with different P')
    plt.plot(points1[:, 0], points1[:, 1], color='r', label='P=8')
    plt.plot(points2[:, 0], points2[:, 1], color='g', label='P=16')
    plt.plot(points3[:, 0], points3[:, 1], color='b', label='P=24')
    plt.plot(points4[:, 0], points4[:, 1], color='m', label='P=32')

    plt.legend()
    plt.savefig('speedup_by_p.png')

    plt.clf()

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    # p=8, r=1
    plt.title('Speedup with different image sizes')
    plt.plot(points5[:, 0], points5[:, 1], color='r', label='Image 250x250')
    plt.plot(points6[:, 0], points6[:, 1], color='g', label='Image 734x694')
    plt.plot(points7[:, 0], points7[:, 1], color='b', label='Image 1920x1080')
    plt.plot(points8[:, 0], points8[:, 1], color='m', label='Image 2048x1536')

    plt.legend()
    plt.savefig('speedup_by_img_sizes.png')

    plt.clf()

    plt.xlabel('Processes')
    plt.ylabel('Speedup')
    plt.axis([1, 17, 1, 3])

    # p=8, r=1, 450x450px
    plt.title('Speedup with datasets of images')
    plt.plot(points9[:, 0], points9[:, 1], color='r', label='10 images')
    plt.plot(points10[:, 0], points10[:, 1], color='g', label='20 images')
    plt.plot(points11[:, 0], points11[:, 1], color='b', label='30 images')
    plt.plot(points12[:, 0], points12[:, 1], color='m', label='40 images')

    plt.legend()
    plt.savefig('speedup_by_ds.png')

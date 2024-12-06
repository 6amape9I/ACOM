import cv2
import numpy as np
from Laba_3.norm_matrix import norm_matrix
from Laba_3.fill_matrix import fill_matrix


def gaus_blur(img, core_size, standard_deviation):
    core = norm_matrix(core_size, fill_matrix(core_size, standard_deviation))

    image = img.copy()

    new_image = np.zeros(shape=(image.shape[0]+12, image.shape[1]+12))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            new_image[i+6][j+6] = image[i][j]


    x_start = core_size // 2
    y_start = core_size // 2
    for i in range(x_start, image.shape[0] - x_start):
        for j in range(y_start, image.shape[1] - y_start):
            # свёртка
            val = 0
            for k in range(-(core_size // 2), core_size // 2 + 1):
                for l in range(-(core_size // 2), core_size // 2 + 1):
                    val += img[i + k, j + l] * core[k + (core_size // 2), l + (core_size // 2)]
            image[i, j] = val

    return image




if __name__ == '__main__':
    core_size = 13
    standard_deviation = 100
    img = cv2.imread(r'./images/img.png', cv2.IMREAD_GRAYSCALE)

    blured_image = gaus_blur(img, core_size, standard_deviation)
    cv2.imshow("Blured", blured_image)

    cv2.imshow('img', img)
    cv2.waitKey(0)

import cv2
import numpy as np


def GAUSBlur(filepath: str, blur_kernel_size: int, scale: float = 1):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return cv2.GaussianBlur(small, (5, 5), cv2.BORDER_DEFAULT)


def svertka(grayscale_image: np.ndarray, kernel: np.ndarray):
    result = np.zeros_like(grayscale_image, np.int32)
    h, w = grayscale_image.shape[:2]
    kernel_size = kernel.shape[0]
    half_kernel_size = int(kernel_size // 2)

    # идем по каждому пикселю изображения без границы
    for y in range(half_kernel_size, h - half_kernel_size):
        for x in range(half_kernel_size, w - half_kernel_size):

            val = 0

            for k in range(-half_kernel_size, half_kernel_size + 1):
                for l in range(-half_kernel_size, half_kernel_size + 1):
                    val += grayscale_image[y + k, x + l] * kernel[half_kernel_size + k, half_kernel_size + l]
            result[y, x] = val

    return result


def angle_value(x, y, tang):
    if (x >= 0 and y <= 0 and tang < -2.414) or (x <= 0 and y <= 0 and tang > 2.414):
        return 0  # сер
    elif x >= 0 and y <= 0 and tang < -0.414:
        return 1  # корич
    elif (x >= 0 and y <= 0 and tang > -0.414) or (x >= 0 and y >= 0 and tang < 0.414):
        return 2  # крас
    elif x >= 0 and y >= 0 and tang < 2.414:
        return 3  # оранж
    elif (x >= 0 and y >= 0 and tang > 2.414) or (x <= 0 and y >= 0 and tang < -2.414):
        return 4  # желт
    elif x <= 0 and y >= 0 and tang < -0.414:
        return 5  # зел
    elif (x <= 0 and y >= 0 and tang > -0.414) or (x <= 0 and y <= 0 and tang < 0.414):
        return 6  # голуб
    elif x <= 0 and y <= 0 and tang < 2.414:
        return 7  # син


def edge_detection(grayscale_image: np.ndarray, lower_bar: float = None, high_bar: float = None,
                   show_grad: bool = True, show_nms: bool = True):
    ker_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ker_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    gx = svertka(grayscale_image, ker_x)
    gy = svertka(grayscale_image, ker_y)

    grad_len = np.sqrt(np.add(np.square(gx), np.square(gy)))
    max_grad_len = grad_len.max()

    if True:
        cv2.imshow('Image gradient', (grad_len / max_grad_len * 255).astype(np.uint8))  # показать градиенты
        cv2.waitKey(0)

    tang = np.divide(gy, gx)
    tang = np.nan_to_num(tang)  # NaN

    print(grad_len)
    print(tang)

    #################################################
    edges = np.zeros_like(grayscale_image)

    for y in range(1, edges.shape[0] - 1):
        for x in range(1, edges.shape[1] - 1):

            angle = angle_value(gx[y, x], gy[y, x], tang[y, x])

            if angle == 0 or angle == 4:
                neighbor1 = [y - 1, x]
                neighbor2 = [y + 1, x]
            elif angle == 1 or angle == 5:
                neighbor1 = [y - 1, x + 1]
                neighbor2 = [y + 1, x - 1]
            elif angle == 2 or angle == 6:
                neighbor1 = [y, x + 1]
                neighbor2 = [y, x - 1]
            elif angle == 3 or angle == 7:
                neighbor1 = [y + 1, x + 1]
                neighbor2 = [y - 1, x - 1]
            else:
                raise Exception('Angle not defined')
            if grad_len[y, x] >= grad_len[neighbor1[0], neighbor1[1]] and grad_len[y, x] > grad_len[
                neighbor2[0], neighbor2[1]]:
                edges[y, x] = 255

    if True:
        cv2.imshow('Image', edges)
        cv2.waitKey(0)




if __name__ == '__main__':
    core_size = 5
    lower_bar = 0.30
    high_bar = 0.40

    path = r'./images/img.png'
    img = GAUSBlur(path, core_size)

    cv2.imshow('img', img)
    edge_detection(img, lower_bar, high_bar, show_grad=True, show_nms=True)

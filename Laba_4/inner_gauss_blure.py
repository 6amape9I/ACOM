import cv2
import numpy as np


def GAUSBlur(filepath: str, blur_kernel_size: int, scale: float = 1):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    return cv2.GaussianBlur(small, (blur_kernel_size, blur_kernel_size), cv2.BORDER_DEFAULT)

if __name__ == '__main__':
    core_size = 5
    path = r'./images/image.png'
    img = GAUSBlur(path, core_size)

    cv2.imshow('img', img)
    cv2.waitKey(0)
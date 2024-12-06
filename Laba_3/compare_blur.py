import cv2
from Laba_3.gaus_blur import gaus_blur


if __name__ == '__main__':
    core_size = 5
    standard_deviation = 10
    img = cv2.imread(r'./images/image.png', cv2.IMREAD_GRAYSCALE)

    blured_image = gaus_blur(img, core_size, standard_deviation)
    cv2.imshow("Blured", blured_image)

    blured_image_cv = cv2.GaussianBlur(img, (core_size, core_size), standard_deviation)
    cv2.imshow("Blured CV", blured_image_cv)

    cv2.imshow('img', img)
    cv2.waitKey(0)

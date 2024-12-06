import cv2


def normal_image_png():
    image_path = r'../Laba_4/images/image.png'
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gray_image_jpg():
    image_path = r'../Laba_4/images/image.jpg'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def any_image_jpeg():
    image_path = r'../Laba_4/images/image.bmp'
    image = cv2.imread(image_path, cv2.COLOR_YCR_CB2RGB)
    cv2.namedWindow('Image', cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    normal_image_png()
    gray_image_jpg()
    any_image_jpeg()

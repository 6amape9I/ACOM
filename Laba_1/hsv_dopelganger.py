import cv2

def hsv_dopelganger():
    im1 = cv2.imread( r'./images/image.png')
    im2 = cv2.imread( r'./images/image.png')

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('hsv image', cv2.WINDOW_NORMAL)

    cv2.imshow('image',im1)

    hsv = cv2.cvtColor(im2, cv2.COLOR_BGR2HSV)
    cv2.imshow('hsv image', hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    hsv_dopelganger()
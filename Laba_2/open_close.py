import cv2
import numpy as np

def open_close():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (80, 10, 0), (100, 255, 255))

        kernel = np.ones((5, 5), np.uint8)

        image_opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        image_closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Open", image_opening)
        cv2.imshow("Close", image_closing)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def erode(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    eroded = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            eroded[i, j] = np.min(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return eroded

def dilate(image, kernel):
    m, n = image.shape
    km, kn = kernel.shape
    hkm = km // 2
    hkn = kn // 2
    dilated = np.copy(image)

    for i in range(hkm, m - hkm):
        for j in range(hkn, n - hkn):
            dilated[i, j] = np.max(
                image[i - hkm:i + hkm + 1, j - hkn:j + hkn + 1][kernel == 1])

    return dilated

if __name__ == '__main__':
    open_close()
    # image = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
    # kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # eroded = erode(image, kernel)
    # dilated = dilate(image, kernel)
    # cv2.imshow("Original", image)
    # cv2.imshow("Eroded", eroded)
    # cv2.imshow("Dilated", dilated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows
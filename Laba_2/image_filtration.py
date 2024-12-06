import cv2
import numpy as np


def blue_color_mask():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #пипетка
        height, width, _ = frame.shape
        center = (width // 2, height // 2)
        size = min(width, height) // 4


        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #central_pixel_color = frame[center[1], center[0]].tolist()
        #print(central_pixel_color)

        lower_blue = np.array([80, 10, 0])
        lupper_blue = np.array([100, 255, 255])
        mask = cv2.inRange(hsv, lower_blue, lupper_blue)
        blue_frame = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]

        cv2.imshow('Blue Mask', blue_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    blue_color_mask()
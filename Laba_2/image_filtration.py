import cv2
import numpy as np

def blue_color_mask():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (80, 10, 0), (100, 255, 255))
        blue_frame = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Blue Mask', blue_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    blue_color_mask()
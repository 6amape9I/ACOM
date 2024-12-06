import cv2
import numpy as np

def find_moment():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (80, 10, 0), (100, 255, 255))


        moments = cv2.moments(mask)
        area = moments['m00']
        print(area)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    find_moment()
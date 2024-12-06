import cv2
import numpy as npq

def rectangle_obj():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, (90, 60, 60), (110, 150, 150))


        moments = cv2.moments(mask)
        area = moments['m00']
        print(area)

        if area > 0:
            width = height = int(np.sqrt(area)/2)
            c_x = int(moments["m10"] / moments["m00"])
            c_y = int(moments["m01"] / moments["m00"])
            cv2.rectangle(frame,
                          (c_x - (width // 12), c_y - (height // 12)),
                          (c_x + (width // 12), c_y + (height // 12)),
                          (0, 0, 0), 2)

        cv2.imshow('Rectanle_frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    rectangle_obj()
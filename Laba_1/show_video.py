import cv2

def show_video():
    capture = cv2.VideoCapture(r'./videos/hello.mp4')

    if not capture.isOpened():
        exit()

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (400, 400))
        hz_frame = cv2.cvtColor(frame, cv2.COLOR_YCR_CB2RGB)

        cv2.imshow('Video 1', frame)
        cv2.imshow('Video 2', small_frame)
        cv2.imshow('Video 3', hz_frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    show_video()
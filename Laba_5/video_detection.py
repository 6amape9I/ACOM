import cv2


def video_detection(kernel_size, standard_deviation, delta_tresh, min_area):
    video = cv2.VideoCapture(0)

    ret, frame = video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(r'videos/output.mp4', fourcc, 144, (w, h))

    while True:
        old_img = img.copy()
        ret, frame = video.read()
        if not ret or cv2.waitKey(1) & 0xFF == ord('q'):
            break


        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), standard_deviation)

        diff = cv2.absdiff(img, old_img)
        thresh = cv2.threshold(diff, delta_tresh, 255, cv2.THRESH_BINARY)[1]
        (contours, hierarchy) = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)

        for contr in contours:
            area = cv2.contourArea(contr)

            if area < min_area:
                continue
            video_writer.write(frame)

    video_writer.release()


if __name__ == '__main__':
    gauss_kernel_size = 3
    gauss_deviation = 50
    delta_trash = 60
    min_area = 20
    video_detection(gauss_kernel_size, gauss_deviation, delta_trash, min_area)

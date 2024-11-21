import cv2

def video_transition():
    video = cv2.VideoCapture('./videos/hello.mp4')
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('./videos/output.mov', fourcc, 25, (w, h))

    while True:
        ok, img = video.read()
        if not ok:
            break
        cv2.imshow('img', img)
        video_writer.write(img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_transition()
import cv2
import numpy as np

def draw_pentagram(image, center, size, color, thickness):
    points = []
    for i in range(5):
        angle = i * 144 * np.pi / 180
        x = int(center[0] + size * np.cos(angle))
        y = int(center[1] - size * np.sin(angle))
        points.append((x, y))

    round_points = []
    for i in range(180):
        angle = i * 2 * np.pi / 180
        x = int(center[0] + size * np.cos(angle))
        y = int(center[1] - size * np.sin(angle))
        round_points.append((x, y))

    for i in range(179):
        cv2.line(image, round_points[i], round_points[i+1], color, thickness)

    for i in range(5):
        cv2.line(image, points[i], points[(i + 1) % 5], color, thickness)

def camera_modif():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, _ = frame.shape
        cross_image = np.zeros((height, width, 3), dtype=np.uint8)

        center = (width // 2, height // 2)
        size = min(width, height) // 4
        color = (0, 0, 255)
        thickness = 2

        draw_pentagram(cross_image, center, size, color, thickness)
        result_frame = cv2.addWeighted(frame, 1, cross_image, 0.5, 0)
        cv2.imshow("Red Pentagram", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    camera_modif()
import cv2
import argparse
import json
import numpy as np
from pathlib import Path

COLOR_MAP = {
    "left": (255, 0, 0),
    "right": (0, 0, 255),
    "ego": (0, 255, 0),
    "other": (0, 255, 255),
}

def scale_lines(points, orig_size=(1024, 512)):
    orig_w, orig_h = orig_size

    scaled = []
    for x, y in points:
        x_px = x * orig_w
        y_px = y * orig_h
        scaled.append((x_px, y_px))

    # Convert to proper format for cv2.polylines
    pts = np.array(scaled, dtype=np.int32)  # shape (N, 2)
    pts = pts.reshape((-1, 1, 2))  # shape (N, 1, 2)

    return pts


def draw_lanes(image_path):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    pts = []

    label_path = Path(image_path.replace("images", "labels").replace(".jpg", ".txt"))
    with label_path.open("r") as f:
        data = json.load(f)

        for item in data:
            xp = np.array(item["xp"], dtype=np.float32) * 1024
            yp = np.linspace(0, 511, 64, dtype=int)
            h_vector = item["h_vector"]
            cls = item["class"]

            for x, y, h in zip(xp, yp, h_vector):
                if h == 1:
                    cv2.circle(image, (int(x), int(y)), 2, COLOR_MAP[cls], thickness=-1)


    # Show image
    cv2.imshow("Image with bbox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def draw_points(image_path):
    image = cv2.imread(image_path)
    img_h, img_w = image.shape[:2]
    pts = []

    label_path = Path(image_path.replace("images", "labels").replace(".jpg", ".txt"))
    with label_path.open("r") as f:
        data = json.load(f)

        for item in data:
            pts = scale_lines(item["points"])
            # h_vector = scale_lines(item["h_vector"])
            cv2.polylines(image, [pts], isClosed=False, color=(255, 0, 0), thickness=3)

    # Show image
    cv2.imshow("Image with bbox", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="image path")
    args = parser.parse_args()

    image = args.image_path
    # draw_bbox(image)
    # draw_lanes(image)
    draw_lanes(image)

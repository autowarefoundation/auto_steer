from argparse import ArgumentParser
import cv2
import numpy as np
from PIL import Image
from Models.inference.auto_steer_infer import AutoSteerNetworkInfer

color_map = {
    1: (0, 0, 255),  # red
    2: (0, 255, 255),  # yellow
    3: (255, 255, 0)  # cyan
}

def make_visualization(img, xp, h_vector):
    yp = np.linspace(0, 511, 64, dtype=int)

    # l_xp = xp[0] * 1024
    # l_h_vector = h_vector[0]
    # l_h_vector = (l_h_vector >= 0.5).astype(int)
    # l_xp = l_xp * l_h_vector

    e_xp = xp * 1024
    e_h_vector = h_vector
    e_h_vector = (e_h_vector >= 0.5).astype(int)
    e_xp = e_xp * e_h_vector

    # r_xp = xp[2] * 1024
    # r_h_vector = h_vector[2]
    # r_h_vector = (r_h_vector >= 0.5).astype(int)
    # r_xp = r_xp * r_h_vector

    # # Left
    # for x, y, h in zip(l_xp, yp, l_h_vector):
    #     if h == 1:
    #         cv2.circle(frame, (int(x), int(y)), 3, (228, 186, 20), thickness=-1)
    # Ego
    for x, y, h in zip(e_xp, yp, e_h_vector):
        if h == 1:
            cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), thickness=-1)
    # Right
    # for x, y, h in zip(r_xp, yp, r_h_vector):
    #     if h == 1:
    #         cv2.circle(frame, (int(x), int(y)), 3, (180,105, 255), thickness=-1)

    cv2.imshow("Visualization", img)
    cv2.waitKey(0)  # waits indefinitely for a key press
    cv2.destroyAllWindows()  # closes the window


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-p", "--model_checkpoint_path", dest="model_checkpoint_path",
                        help="path to pytorch checkpoint file to load model dict")
    parser.add_argument("-i", "--input_image_filepath", dest="input_image_filepath",
                        help="path to input image which will be processed by DomainSeg")
    args = parser.parse_args()
    model_checkpoint_path = args.model_checkpoint_path
    input_image_filepath = args.input_image_filepath

    model = AutoSteerNetworkInfer(model_checkpoint_path)
    img = cv2.imread(input_image_filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xp, h_vector = model.inference(img)
    make_visualization(img, xp, h_vector)
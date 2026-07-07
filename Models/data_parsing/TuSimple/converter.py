#! /usr/bin/env python3

import os
import cv2
import json
import math
import argparse
import warnings
import numpy as np
import shutil
import random
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw

# Custom warning format cuz the default one is wayyyyyy too verbose
def custom_warning_format(message, category, filename, lineno, line=None):
    return f"WARNING : {message}\n"


warnings.formatwarning = custom_warning_format


# ============================== Helper functions ============================== #


def round_points(line, ndigits=4):
    line = list(line)
    for i in range(len(line)):
        line[i] = [
            round(line[i][0], ndigits),
            round(line[i][1], ndigits)
        ]
    line = tuple(line)
    return line


def sortByYDesc(line):
    return sorted(
        line,
        key=lambda p: p[1],
        reverse=True
    )


def scale_points(points, orig_size=(1280, 720), crop_top=80, new_size=(1024, 512)):
    orig_w, orig_h = orig_size
    new_w, new_h = new_size

    # After cropping
    cropped_h = orig_h - crop_top

    # Scale factors
    scale_x = new_w / orig_w
    scale_y = new_h / cropped_h

    transformed = []
    for x, y in points:
        # 1. Crop (shift y)
        y_new = y - crop_top

        # Optional: skip points that were in the cropped region
        if y_new < 0:
            continue

        # 2. Scale
        x_new = x * scale_x
        y_new = y_new * scale_y

        transformed.append((x_new, y_new))

    return transformed


def normalize_points(lane, width, height):
    """
    Normalize the coords of lane points.

    """
    return [(x / width, y / height) for x, y in lane]


def getLineAnchor(line, verbose=False):
    """
    Determine "anchor" point of a lane.

    """
    (x2, y2) = line[0]
    (x1, y1) = line[1]

    for i in range(1, len(line) - 1, 1):
        if (line[i][0] != x2) & (line[i][1] != y2):
            (x1, y1) = line[i]
            break

    if (x1 == x2) or (y1 == y2):
        if (x1 == x2):
            error_lane = "Vertical"
        elif (y1 == y2):
            error_lane = "Horizontal"
        if (verbose):
            warnings.warn(f"{error_lane} line detected: {line}, with these 2 anchors: ({x1}, {y1}), ({x2}, {y2}).")
        return (x1, None, None)

    a = (y2 - y1) / (x2 - x1)
    b = y1 - a * x1
    x0 = (H - 1 - b) / a

    return (x0, a, b)


def getEgoIndexes(anchors):
    """
    Identifies 2 ego lanes - left and right - from a sorted list of lane anchors.

    """
    for i in range(len(anchors)):
        if (anchors[i][0] >= W / 2):
            if (i == 0):
                return "NO LANES on the LEFT side of frame. Something's sussy out there!"
            left_ego_idx, right_ego_idx = i - 1, i
            return (left_ego_idx, right_ego_idx)

    return "NO LANES on the RIGHT side of frame. Something's sussy out there!"


def getDrivablePath(left_ego, right_ego):
    """
    Computes drivable path as midpoint between 2 ego lanes, basically the main point of this task.

    """
    i, j = 0, 0
    drivable_path = []
    while (i < len(left_ego) - 1 and j < len(right_ego) - 1):
        if (left_ego[i][1] == right_ego[j][1]):
            drivable_path.append((
                (left_ego[i][0] + right_ego[j][0]) / 2,  # Midpoint along x axis
                left_ego[i][1]
            ))
            i += 1
            j += 1
        elif (left_ego[i][1] < right_ego[j][1]):
            i += 1
        else:
            j += 1

    # Extend drivable path to bottom edge of the frame
    if (len(drivable_path) >= 2):
        x1, y1 = drivable_path[-2]
        x2, y2 = drivable_path[-1]
        if (x2 == x1):
            x_bottom = x2
        else:
            a = (y2 - y1) / (x2 - x1)
            x_bottom = x2 + (H - y2) / a
        drivable_path.append((x_bottom, H))

    # Extend drivable path to be on par with longest ego lane
    # By making it parallel with longer ego lane
    y_top = min(left_ego[0][1], right_ego[0][1])
    sign_left_ego = left_ego[0][0] - left_ego[1][0]
    sign_right_ego = right_ego[0][0] - right_ego[1][0]
    sign_val = sign_left_ego * sign_right_ego
    if (sign_val > 0):  # 2 egos going the same direction
        longer_ego = left_ego if left_ego[0][1] < right_ego[0][1] else right_ego
        if len(longer_ego) >= 2 and len(drivable_path) >= 2:
            x1, y1 = longer_ego[0]
            x2, y2 = longer_ego[1]
            if (x2 == x1):
                x_top = drivable_path[0][0]
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = drivable_path[0][0] + (y_top - drivable_path[0][1]) / a

            drivable_path.insert(0, (x_top, y_top))
    else:
        # Extend drivable path to be on par with longest ego lane
        if len(drivable_path) >= 2:
            x1, y1 = drivable_path[0]
            x2, y2 = drivable_path[1]
            if (x2 == x1):
                x_top = x1
            else:
                a = (y2 - y1) / (x2 - x1)
                x_top = x1 + (y_top - y1) / a

            drivable_path.insert(0, (x_top, y_top))

    return drivable_path


def annotateGT(anno_entry, anno_raw_file, raw_dir, mask_dir, visualization_dir, ):
    """
    Annotates and saves an image with:
        - Raw image, in "output_dir/image".
        - Lane seg mask, in "output_dir/mask".
        - Annotated image with all lanes, in "output_dir/visualization".

    """

    # Load raw image
    # raw_img = Image.open(anno_raw_file).convert("RGB")

    # Define save name
    save_name = str(img_id_counter).zfill(6)

    # Copy raw img and put it in raw dir.
    shutil.copy(anno_raw_file, os.path.join(raw_dir, save_name + ".jpg"))

    # # Save mask, lossless PNG for accuracy
    # mask_img = Image.fromarray(anno_entry["mask"]).convert("RGB")
    # mask_img.save(os.path.join(mask_dir, save_name + ".png"))
    #
    # # Overlay mask on raw image, ratio 1:1
    # overlayed_img = Image.blend(raw_img, mask_img, alpha=0.5)
    #
    # # Save visualization img, JPG for lighter weight, just different dir
    # overlayed_img.save(os.path.join(visualization_dir, save_name + ".jpg"))


def calcLaneSegMask(lanes, width, height, normalized: bool = True):
    """
    Calculates binary segmentation mask for some lane lines.

    """

    # Create blank mask as new Image
    bin_seg = np.zeros(
        (height, width),
        dtype=np.uint8
    )
    bin_seg_img = Image.fromarray(bin_seg)

    # Draw lines on mask
    draw = ImageDraw.Draw(bin_seg_img)
    for lane in lanes:
        if (normalized):
            lane = [
                (
                    x * width,
                    y * height
                )
                for x, y in lane
            ]
        draw.line(
            lane,
            fill=255,
            width=4
        )

    # Convert back to numpy array
    bin_seg = np.array(
        bin_seg_img,
        dtype=np.uint8
    )

    return bin_seg


def convert_image(input_img, output_img):
    # Load image
    img = cv2.imread(input_img)  # replace with your path

    # Check if loaded
    if img is None:
        raise ValueError("Image not found or path is incorrect")

    # Crop 80px from the top (keep bottom part)
    cropped = img[80:, :]

    # Resize from (1280x640) → (1024x512)
    resized = cv2.resize(cropped, (1024, 512), interpolation=cv2.INTER_LINEAR)

    # Save or display
    cv2.imwrite(output_img, resized)
    # cv2.imshow("result", resized); cv2.waitKey(0)


def create_h_vector(points):
    image_height = 512
    num_rows = 64
    row_height = image_height // num_rows  # 8 pixels per row

    # Initialize binary vector of zeros
    h_vector = np.zeros(num_rows, dtype=int)
    min_y_row = min(p[1] for p in points) // row_height
    max_y_row = max(p[1] for p in points) // row_height

    # Loop through points and mark the corresponding row
    for i in range(num_rows):
        if min_y_row <= i <= max_y_row:
            h_vector[i] = 1

    return h_vector


def compute_center_line(left, right):
    left = np.array(left)
    right = np.array(right)

    # Sort by Y
    left = left[np.argsort(left[:, 1])]
    right = right[np.argsort(right[:, 1])]

    # Overlapping Y range
    y_top = max(left[:, 1].min(), right[:, 1].min())
    y_bottom = min(left[:, 1].max(), right[:, 1].max())

    if y_bottom < y_top:
        return None

    # Use ALL available Y values from both lanes
    y_left = left[:, 1]
    y_right = right[:, 1]

    y_all = np.concatenate([y_left, y_right])

    # Keep only overlapping region
    y_all = y_all[(y_all >= y_top) & (y_all <= y_bottom)]

    # Remove duplicates + sort
    y_samples = np.unique(y_all)

    # Interpolate X
    left_x = np.interp(y_samples, left[:, 1], left[:, 0])
    right_x = np.interp(y_samples, right[:, 1], right[:, 0])

    # Center
    center_x = (left_x + right_x) / 2.0

    center_line = np.stack([center_x, y_samples], axis=1)
    return center_line


def sample_points(points):
    points = np.array(points, dtype=np.float32)

    # sort by y (VERY important)
    points = points[np.argsort(points[:, 1])]

    x, y = points[:, 0], points[:, 1]
    # mask valid region
    min_y, max_y = y.min(), y.max()

    # fit lower degree polynomial (more stable)
    coeffs = np.polyfit(y, x, 5)  # try 2 or 3, not 5
    p = np.poly1d(coeffs)

    # uniform sampling along y
    yp = np.linspace(0, 511, 64)
    xp = p(yp)
    invalid = (yp < min_y) | (yp > max_y)
    xp[xp < 0] = 0
    # set corresponding xp to 0
    xp[invalid] = 0


    valid = (yp >= min_y) & (yp <= max_y)
    h_vector = np.zeros(64)
    h_vector[valid] = 1

    return xp, h_vector


def convert_label(points, cls):
    points = scale_points(points)

    xp, h_vector = sample_points(points)

    # h_vector = create_h_vector(points)
    # points = round_points(normalize_points(points, 1024, 512))

    label = {
        "class": cls,
        "xp": (xp / 1024).tolist(),
        "h_vector": h_vector.tolist(),
    }

    return label


def convert_labels(item: dict, verbose: bool = False):
    W = 1024
    H = 512

    lanes = item["lanes"]
    h_samples = item["h_samples"]
    raw_file = item["raw_file"]

    # Decouple from {lanes: [xi1, xi2, ...], h_samples: [y1, y2, ...]} to [(xi1, y1), (xi2, y2), ...]
    # `lane_decoupled` is a list of sublists representing lanes, each lane is a list of (x, y) tuples.
    lanes_decoupled = [
        [(x, y) for x, y in zip(lane, h_samples) if x != -2]
        for lane in lanes if sum(1 for x in lane if x != -2) >= 2
        # Filter out lanes < 2 points (there's actually a bunch of em)
    ]

    # Sort each lane by decreasing y
    lanes_decoupled = [
        sortByYDesc(lane)
        for lane in lanes_decoupled
    ]

    # Determine 2 ego lanes
    lane_anchors = [
        getLineAnchor(lane)
        for lane in lanes_decoupled
    ]
    ego_indexes = getEgoIndexes(lane_anchors)

    if (type(ego_indexes) is str):
        if (ego_indexes.startswith("NO")):
            if (verbose):
                warnings.warn(f"Parsing {raw_file}: {ego_indexes}")
            return None

    left_line = lanes_decoupled[ego_indexes[0]]
    right_line = lanes_decoupled[ego_indexes[1]]
    other_lines = [
        lane for idx, lane in enumerate(lanes_decoupled)
        if idx not in ego_indexes
    ]

    # Parse processed data, all coords normalized
    labels = []

    # left line
    left_line_label = convert_label(left_line, "left")
    labels.append(left_line_label)

    # right line
    right_line_label = convert_label(right_line, "right")
    labels.append(right_line_label)

    # ego line
    ego_line_points = compute_center_line(left_line, right_line)
    ego_line_label = convert_label(ego_line_points, "ego")
    labels.append(ego_line_label)

    # other lanes
    for line in other_lines:
        labels.append(convert_label(line, "other"))

    return labels

def expand_training_set(dataset_dir, fract=0.25):
    val_images_dir = dataset_dir + "/images/val"
    val_labels_dir = dataset_dir + "/labels/val"

    train_images_dir = dataset_dir + "/images/train"
    train_labels_dir = dataset_dir + "/labels/train"

    val_images = [f for f in Path(val_images_dir).rglob("*") if f.is_file()]
    random.shuffle(val_images)

    split_idx = int(len(val_images) * fract)
    # val_files = files[:split_idx]
    train_images = val_images[split_idx:]

    for image in tqdm(train_images, desc="Expand training dataset", unit="file"):
        if image.is_file():
            target_image = Path(train_images_dir) / image.name
            label = Path(val_labels_dir) / f"{image.stem}.txt"
            target_label = Path(train_labels_dir) / f"{image.stem}.txt"

            # Move file
            try:
                shutil.move(str(image), str(target_image))
                shutil.move(str(label), str(target_label))
            except Exception as e:
                print(f"Failed to move {image.name}: {e}")


if __name__ == "__main__":
    root_dir = ""
    train_dir = "train_set"
    test_dir = "test_set"

    train_clip_codes = ["0313", "0531", "0601"]  # Train labels are split into 3 dirs
    test_file = "test_label.json"  # Test file name

    W = 1280
    H = 720
    # ============================== Parsing args ============================== #

    parser = argparse.ArgumentParser(description="Process TuSimple dataset - PathDet groundtruth generation")
    parser.add_argument("--dataset_dir", "-i", type=str, help="TuSimple directory (right after extraction)")
    parser.add_argument("--output_dir", "-o", type=str, help="Output directory")
    # For debugging only
    parser.add_argument("--early_stopping", "-e", type=int,
                        help="Num. files each split/class you wanna limit, instead of whole set.", required=False)
    args = parser.parse_args()

    # Generate output structure
    """
    --output_dir
        |----image
        |----mask
        |----visualization
        |----drivable_path.json
    """
    dataset_dir = args.dataset_dir
    output_dir = args.output_dir

    if (os.path.exists(output_dir)):
        print(f"Output dir {output_dir} already exists, purged!")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # list_subdirs = ["image", "mask", "visualization"]
    # for subdir in list_subdirs:
    #     subdir_path = os.path.join(output_dir, subdir)
    #     if (not os.path.exists(subdir_path)):
    #         os.makedirs(subdir_path, exist_ok=True)

    # Parse early stopping
    if (args.early_stopping):
        print(f"Early stopping set, stops after {args.early_stopping} files.")
        early_stopping = args.early_stopping
    else:
        early_stopping = None

    # ============================== Parsing annotations ============================== #

    train_label_files = [
        os.path.join(dataset_dir, root_dir, train_dir, f"label_data_{clip_code}.json")
        for clip_code in train_clip_codes
    ]

    """
    Make no mistake: the file `TuSimple/test_set/test_tasks_0627.json` is NOT a label file.
    It is just a template for test submission. Kinda weird they put it in `test_set` while
    the actual label file is in `TuSimple/test_label.json`.
    """
    test_label_files = [os.path.join(dataset_dir, root_dir, test_file)]
    label_files = train_label_files + test_label_files

    # Parse data by batch
    data_master = {}

    img_id_counter = -1

    for anno_file in label_files:

        print(f"\n==================== Processing data in label file {anno_file} ====================\n")

        # Read each line of GT text file as JSON
        with open(anno_file, "r") as f:
            read_data = [json.loads(line) for line in f.readlines()]

        for frame_data in tqdm(read_data, desc=f"Parsing annotations of {os.path.basename(anno_file)}", colour="green"):
            # Conduct index increment
            img_id_counter += 1

            # Early stopping, it defined
            if (early_stopping and img_id_counter >= early_stopping - 1):
                break

            # set_dir = "/".join(anno_file.split("/")[ : -1]) # Slap "train_set" or "test_set" to the end  <-- Specific to linux hence used os.path.dirname command below
            set_dir = os.path.dirname(anno_file)
            set_dir = os.path.join(set_dir, test_dir) if test_file in anno_file else set_dir  # Tricky test dir
            split = "val" if test_file in anno_file else "train"

            # Parse annotations
            raw_file = frame_data["raw_file"]
            anno_entry = convert_labels(frame_data)
            if (anno_entry is None):
                continue

            # # Annotate raw images
            # annotateGT(
            #     anno_entry,
            #     anno_raw_file=os.path.join(set_dir, raw_file),
            #     raw_dir=os.path.join(output_dir, "image"),
            #     mask_dir=os.path.join(output_dir, "mask"),
            #     visualization_dir=os.path.join(output_dir, "visualization")
            # )

            file_name = str(img_id_counter).zfill(6)
            # Save image
            input_img = os.path.join(set_dir, raw_file)
            output_images_dir = os.path.join(output_dir, "images", split)
            output_img = os.path.join(output_images_dir, file_name + ".jpg")
            os.makedirs(output_images_dir, exist_ok=True)

            convert_image(input_img, output_img)

            # shutil.copy(input_img, output_img)

            # Save label
            label = os.path.join(output_dir, "labels", split, file_name + ".txt")
            os.makedirs(os.path.dirname(label), exist_ok=True)
            with open(label, "w") as f:
                json.dump(anno_entry, f, indent=4)

            # # Change `raw_file` to 6-digit incremental index
            # data_master[str(img_id_counter).zfill(6)] = {
            #     # "drivable_path" : anno_entry["drivable_path"],
            #     "egoleft_lane": anno_entry["egoleft_lane"],
            #     "egoright_lane": anno_entry["egoright_lane"],
            #     "other_lanes": anno_entry["other_lanes"],
            # }

        print(f"Processed {len(read_data)} entries in above file.\n")

    expand_training_set(output_dir)

    print(f"Done processing data with {len(data_master)} entries in total.\n")

    # # Save master data
    # with open(os.path.join(output_dir, "drivable_path.json"), "w") as f:
    #     json.dump(data_master, f, indent=4)

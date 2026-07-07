import os
import json
import argparse
import random
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm  # pip install tqdm
from pathlib import Path

orig_image_width = 2560
orig_image_height = 1440
image_width = 1024
image_height = 512
crop_size = 160


def convert_file_paths(input_file_paths, subdirectory, extension):
    target_file_paths = []
    for file in input_file_paths:
        p = Path(file)

        # Replace folder name
        parts = list(p.parts)
        parts[parts.index("labels")] = subdirectory
        parts[-1] = parts[-1].replace(".lines.json", extension)
        target_file_path = Path(*parts)

        # Change extension
        target_file_path = target_file_path.with_suffix(extension)

        target_file_paths.append(target_file_path)

    return target_file_paths


def convert_images(image_file_paths, output_dir):
    output_dir = Path(output_dir)

    for file in tqdm(image_file_paths, desc="Convert images", unit="file"):
        try:
            with Image.open(file) as img:
                width, height = img.size
                # Crop: (left, upper, right, lower)
                cropped = img.crop((0, 160, width, height))
                # Resize
                resized = cropped.resize((1024, 512), Image.LANCZOS)

                # Save to output directory
                target = output_dir / file.name
                target.parent.mkdir(parents=True, exist_ok=True)

                # If file with same name exists, rename
                if target.exists():
                    print(f"File with same name already exists: {target}")
                    # optional: add suffix
                    stem, ext = file.stem, file.suffix
                    target = output_dir / f"{stem}_cropped{ext}"

                with open(target, "wb") as f:
                    resized.save(f)
                    f.flush()  # flush Python internal buffers
                    os.fsync(f.fileno())
        except Exception as e:
            print(f"Failed to process {file}: {e}")


def sort_lines(lines):
    x_center = image_width / 2.0

    left_line = None
    right_line = None
    left_line_dist = float('inf')
    right_line_dist = float('inf')
    other_lines = []

    for line in lines:
        classified = False
        if line.shape[0] < 2:
            continue

        # Find lowest point (max y)
        idx_lowest = np.argmax(line[:, 1])
        x_lowest = line[idx_lowest, 0]
        y_lowest = line[idx_lowest, 1]

        dist_to_center = abs(x_lowest - x_center)

        # Left candidate
        if x_lowest < x_center:
            if dist_to_center < left_line_dist:
                left_line_dist = dist_to_center
                left_line = line
                classified = True

        # Right candidate
        elif x_lowest > x_center:
            if dist_to_center < right_line_dist:
                right_line_dist = dist_to_center
                right_line = line
                classified = True

        if not classified:
            other_lines.append(line)

    if left_line is not None:
        left_line = left_line[np.argsort(-left_line[:, 1])]
    if right_line is not None:
        right_line = right_line[np.argsort(-right_line[:, 1])]
    if len(other_lines) > 0:
        other_lines = [
            line[np.argsort(-line[:, 1])] for line in other_lines
        ]

    return left_line, right_line, other_lines


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
    xp, h_vector = sample_points(points)

    # h_vector = create_h_vector(points)
    # points = round_points(normalize_points(points, 1024, 512))

    label = {
        "class": cls,
        "xp": (xp / 1024).tolist(),
        "h_vector": h_vector.tolist(),
    }

    return label


def find_center_line(left_line, right_line):
    # not enough points
    if len(left_line) < 3 or len(right_line) < 3:
        return np.empty((0, 2), dtype=np.int32)

    # Sort by y
    left_line = left_line[np.argsort(left_line[:, 1])]
    right_line = right_line[np.argsort(right_line[:, 1])]

    # Compute overlapping y-range
    y_min = max(left_line[:, 1].min(), right_line[:, 1].min())
    y_max = min(left_line[:, 1].max(), right_line[:, 1].max())

    mask = (left_line[:, 1] >= y_min) & (left_line[:, 1] <= y_max)
    left_crop = left_line[mask]

    if len(left_crop) < 3:
        return []

    # Interpolate right lane at left y positions
    interp_right_x = np.interp(
        left_crop[:, 1],
        right_line[:, 1],
        right_line[:, 0]
    )

    center_x = (left_crop[:, 0] + interp_right_x) / 2
    center_y = left_crop[:, 1]

    center_line = np.stack([center_x, center_y], axis=1)

    # Fit polynomial using lower region only (stable)
    y_vals = center_line[:, 1]
    x_vals = center_line[:, 0]

    bottom_mask = y_vals > np.percentile(y_vals, 70)

    if np.sum(bottom_mask) < 3:
        bottom_mask = np.ones_like(y_vals, dtype=bool)

    poly = np.polyfit(y_vals[bottom_mask], x_vals[bottom_mask], 2)

    # Extend to bottom of image
    y_start = int(y_vals.max())
    y_end = image_height - 1

    y_extended = np.linspace(y_start, y_end, 100)
    x_extended = np.polyval(poly, y_extended)

    extended_part = np.stack([x_extended, y_extended], axis=1)

    center_line = np.vstack([center_line, extended_part])

    # Keep only valid image points
    center_line = center_line[(center_line[:, 0] >= 0) & (center_line[:, 0] < image_width)]

    center_line = np.round(center_line).astype(np.int32)

    return center_line


def convert_labels(input_dir, output_dir):
    input_dir = Path(input_dir)

    files = [f for f in input_dir.rglob("*") if f.is_file()]
    for file in tqdm(files.copy(), desc="Convert labels_path", unit="file"):
        base_name = file.name.split(".", 1)[0]
        scaled_lines_pts = []
        with file.open("rb") as f:
            lines = json.load(f)["Lines"]
            for lane_line in lines:
                # pts = np.array(lane_line).T.astype(np.float32)
                pts_org = np.array([[float(p["x"]), float(p["y"])] for p in lane_line], dtype=np.float32)
                pts = np.copy(pts_org)

                pts[:, 0] = pts_org[:, 0] / (orig_image_width - 1) * (image_width - 1)
                pts[:, 1] = (pts_org[:, 1] - crop_size) / (orig_image_height - crop_size - 1) * (
                        image_height - 1)

                scaled_lines_pts.append(pts)

            # Identify left/right line
        left_line, right_line, other_lines = sort_lines(scaled_lines_pts)

        if left_line is None or right_line is None:
            print("Missing left or right lane: " + base_name)
            files.remove(file)
            continue

        center_line = find_center_line(left_line, right_line)

        # visualize_labels(left_line, right_line, center_line, other_lines)
        # draw_lanes(file, left_line, right_line, center_line, other_lines)

        # Parse processed data, all coords normalized
        labels = []

        if (len(left_line) == 0 or not left_line.shape[1] == 2
                or len(right_line) == 0 or not right_line.shape[1] == 2
                or len(center_line) == 0 or not center_line.shape[1] == 2):
            files.remove(file)
            continue

        # left line
        left_line_label = convert_label(left_line, "left")
        labels.append(left_line_label)

        # right line
        right_line_label = convert_label(right_line, "right")
        labels.append(right_line_label)

        # ego line
        ego_line_label = convert_label(center_line, "ego")
        labels.append(ego_line_label)

        # other lanes
        for line in other_lines:
            labels.append(convert_label(line, "other"))

        target = Path(output_dir) / f"{base_name}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)

        with target.open("w", encoding="utf-8") as f:
            json.dump(labels, f, indent=4)

    return files


def convert(input_ds_dir, output_ds_dir):
    # Convert training data
    # Convert line labels
    input_dir = input_ds_dir + "/train/labels"
    output_dir = output_ds_dir + "/labels/train"
    train_label_file_paths = convert_labels(input_dir, output_dir)

    # Convert images
    train_image_file_paths = convert_file_paths(train_label_file_paths, "images", ".jpg")
    output_training_dir = output_ds_dir + "/images/train"
    convert_images(train_image_file_paths, output_training_dir)


    # convert validation data
    # Convert line labels
    input_dir = input_ds_dir + "/valid/labels"
    output_dir = output_ds_dir + "/labels/val"
    val_label_file_paths = convert_labels(input_dir, output_dir)

    # Convert images
    val_image_file_paths = convert_file_paths(val_label_file_paths, "images", ".jpg")
    output_dir = output_ds_dir + "/images/val"
    convert_images(val_image_file_paths, output_dir)




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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ds_dir", help="Input dataset directory")
    parser.add_argument("-o", "--output_ds_dir", help="Output dataset directory")
    args = parser.parse_args()

    input_ds_dir = args.input_ds_dir
    output_ds_dir = args.output_ds_dir
    convert(input_ds_dir, output_ds_dir)
    expand_training_set(output_ds_dir)

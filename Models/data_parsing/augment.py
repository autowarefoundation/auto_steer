import argparse
import json
from PIL import Image
from pathlib import Path
from tqdm import tqdm  # pip install tqdm


def swap_class(cls):
    if cls == "left":
        return "right"
    elif cls == "right":
        return "left"
    else:
        return cls

def augment(images_dir, labels_dir, desc=""):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    files = [f.name.split(".", 1)[0] for f in images_dir.rglob("*") if f.is_file()]
    for file in tqdm(files.copy(), desc=f"Augment {desc} data", unit="file"):
        # Images
        image = Image.open(images_dir / (file + ".jpg"))
        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped.save(images_dir / (file + "_1" + ".jpg"))

        # Labels
        label = labels_dir / (file + ".txt")
        augmented_labels = []
        with label.open("r") as f:
            data = json.load(open(label, "r"))
            for lane in data:
                augmented_lane = lane.copy()
                xp = lane["xp"]
                h_vector = lane["h_vector"]

                augmented_lane["xp"] = [
                    (1.0 - x) if h == 1.0 else 0.0
                    for x, h in zip(xp, h_vector)
                ]

                # Swap class
                augmented_lane["class"] = swap_class(lane["class"])

                augmented_labels.append(augmented_lane)

        # Save new file
        output_path = Path(labels_dir / (file + "_1" + ".txt"))
        with open(output_path, "w") as f:
            json.dump(augmented_labels, f, indent=4)


def augment_ds(dataset_dir):
    train_images_dir = dataset_dir + "/images/train"
    train_labels_dir = dataset_dir + "/labels/train"
    augment(train_images_dir, train_labels_dir, desc="train")

    val_images_dir = dataset_dir + "/images/val"
    val_labels_dir = dataset_dir + "/labels/val"
    augment(val_images_dir, val_labels_dir, desc="val")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_ds_dir", help="Input dataset directory")
    parser.add_argument("-o", "--output_ds_dir", help="Output dataset directory")
    args = parser.parse_args()

    input_ds_dir = args.input_ds_dir
    output_ds_dir = args.output_ds_dir
    augment_ds(output_ds_dir)

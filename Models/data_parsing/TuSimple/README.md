# process_tusimple.py

## TuSimple dataset preprocessing script for PathDet.

This script parse the [TuSimple lane detection dataset](https://www.kaggle.com/datasets/manideep1108/tusimple?resource=download) (24GB) to create a dataset comprising input images in PNG format and a single drivable path as the ground truth, derived as the mid-line between the left/right ego lanes.

- Data acquisition: by TuSimple - an autonomous trucking company, with 6,408 road images on US highways.
- Features:
    - Different conditions (weather, light, highway, traffic, etc.)
    - Lane detection competition in CVPR 2017 WAD.
- Annotation method: polylines for lane markings

- TuSimple directory structure (from download)
    - `clips/` : video clips
    - `some_clip/` : sequential images, 20 frames
    - `tasks.json` : label data in training set, and a submission template for testing set.

- Data label format:
    - `raw_file` : (str) 20th frame file path in a clip
    - `lanes` : (list) lists of lanes, each list represents a lane. Each element is X-coordinate across the polyline.
    - `h_samples` : (list) list of Y-coordinate across the polyline.
        - `-2` means there is no existing lane marking.
    - Normally each frame has 4 lanes (ego x 2, left, right), but some has 5 (changing lane).


For the purpose of `AutoSteer 2.0` detection OpenLane dataset is preprocessed and directory structure is converted into the following format

```
│          
├─images
│  │  
│  ├─train
│  │      000000.jpg
│  │      ...
│  │      
│  └─val
│          003627.txt
│          ...
│          
└─labels
    │  
    ├─train
    │      000000.jpg
    │      ...
    │      
    └─val
            003627.txt
            ...
```

where annotations are in the following format

```
[
    {
        "class": "left",
        "xp": [
            ....
        ],
        "h_vector": [
            ....    
        ]
    },
    {
        "class": "right",
        "xp": [
        ],
        "h_vector": [
            ....    
        ]
    },
    {
        "class": "ego",
        "xp": [
        ],
        "h_vector": [
            ....    
        ]
    },
    {
        "class": "other",
        "xp": [
        ],
        "h_vector": [
            ....    
        ]
    },
    ...
```

In order to trigger conversion

```
$ python3 converter.py -i <INPUT DATASET DIRECTORY> -o <OUTPUT DATASET DIRECTORY>
```

Dataset conversion can be tested by using test script like:

```
$ python3 test_conversion.py -i <PATH TO THE IMAGE IN DATASET>
```




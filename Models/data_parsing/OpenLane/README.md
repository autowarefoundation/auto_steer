# OpenLane-V1

OpenLane is scaled 3D lane dataset. Dataset collects contents from public perception dataset, providing lane and closest-in-path object(CIPO) annotations for 1000 segments. OpenLane owns 200K frames and over 880K carefully annotated lanes. In v1.0, OpenLane contains the annotations on Waymo Open Dataset.

OpenLane-V dataset labels are used with subset of images from the original dataset. OpenLane-V labels are store in a .pickle file format.

For the purpose of `AutoSteer 2.0` detection OpenLane dataset is preprocessed and directory structure is converted into the following format

```
│          
├─images
│  │  
│  ├─train
│  │      150690619301562700.jpg
│  │      ...
│  │      
│  └─val
│          150690619351536000.txt
│          ...
│          
└─labels
    │  
    ├─train
    │      150690619301562700.jpg
    │      ...
    │      
    └─val
            150690619351536000.txt
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
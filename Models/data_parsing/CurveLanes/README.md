# CurveLanes

CurveLanes is a lane detection dataset with 150K lanes images for difficult scenarios such as curves and multi-lanes in traffic lane detection. It is collected in real urban and highway scenarios in multiple cities in China.

The whole dataset 150K is separate into three parts: train:100K, val: 20K and testing: 30K. The resolution of most images in this dataset is 2650×1440.

For each image, all lanes in image are manually annotated with natural cubic splines. All images are carefully selected so that most of them image contains at least one curve lane. More difficult scenarios such as S-curves, Y-lanes, night and multi-lanes (the number of lane lines is more than 4) can be found in this dataset.

Dataset is organized as:

```
│  
├─test
│  └─images
│          0001bca638957d305a25dd6be2fd5224.jpg
│          ...
│          
├─train
│  │  train.txt
│  │  
│  ├─images
│  │      00007f3230d35a893230c1b3ef8c52fd.jpg
│  │      ...
│  │      
│  └─labels
│          00007f3230d35a893230c1b3ef8c52fd.lines.json
│          ...
│          
└─valid
    │  valid.txt
    │  
    ├─images
    │      00022953ff37d3174cff99833df8799e.jpg
    │      ...
    │      
    └─labels
            00022953ff37d3174cff99833df8799e.lines.json
            ...
```

where annotations are in the following format:

```
{
  "Lines":[
    # A lane marking
    [
      # The x, y coordinates for key points of a lane marking that has at least two key points.
      {
        "y":"1439.0",
        "x":"2079.41"
      },
      {
        "y":"1438.08",
        "x":"2078.19"
      },
      ...
    ]
    ...
  ]
}
```

This dataset organization is converted to a more YOLO like structure

```
│          
├─images
│  │  
│  ├─train
│  │      00007f3230d35a893230c1b3ef8c52fd.jpg
│  │      ...
│  │      
│  └─val
│          00007f3230d35a893230c1b3ef8c52fd.txt
│          ...
│          
└─labels
    │  
    ├─train
    │      00022953ff37d3174cff99833df8799e.jpg
    │      ...
    │      
    └─val
            00022953ff37d3174cff99833df8799e.txt
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
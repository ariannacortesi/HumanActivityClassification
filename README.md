# TP1 – Human Activity Classification, Part A

`mainActivity.py` analyses wearable sensor recordings from the FORTH-TRACE dataset. It detects outliers, uses a linear model to replace injected outliers, and compares PCA, Fisher Score and ReliefF for feature reduction.

## Prepare the data

1. Download the [official FORTH-TRACE dataset](https://github.com/spl-icsforth/FORTH_TRACE_DATASET) using **Code → Download ZIP**.
2. Extract the archive. Place the folder containing `part0` through `part14` next to `mainActivity.py` and name it `FORTH_TRACE_DATASET-master`.

For example, the first recording should be at the following location:

```text
TP1/
├── mainActivity.py
├── README.md
└── FORTH_TRACE_DATASET-master/
    ├── part0/
    │   ├── part0dev1.csv
    │   └── ...
    ├── ...
    └── part14/
```

## Run Part A

Open a terminal in the folder containing `mainActivity.py`. Install the dependencies once:

```sh
python -m pip install numpy pandas scipy scikit-learn "matplotlib>=3.9"
```

Run the script:

```sh
python mainActivity.py
```

The script processes all 75 recordings. It prints the results in the terminal and displays the plots at the end.

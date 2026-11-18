# TP1 – Human Activity Classification

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

## Run the script

Open a terminal in the folder containing `mainActivity.py`. Install the dependencies once:

```sh
python -m pip install numpy pandas scipy scikit-learn "matplotlib>=3.9"
```

Run the script:

```sh
python mainActivity.py
```

The script runs Part A followed by exercise B.1. It processes all 75 recordings, prints the results in the terminal and displays the plots at the end. The `main()` function calls `part_a()` and `part_b()` in this order.

## Part B - Exercise 1

The Part B section of `mainActivity.py` implements Train-Test, Train-Validation-Test and K-fold splitting, followed by confusion matrix, recall, precision and F1-score calculations. Train-Test and Train-Validation-Test preserve class proportions. Metrics use macro averaging, giving each class equal importance.

The example uses Iris to demonstrate the splits and a small set of known predictions to demonstrate the metrics.

## Implementation choices

The group implementations are combined in `mainActivity.py`. K-means and PCA use explicit NumPy steps. The original versions remain available in the Git history.

- Z-score outliers are calculated separately for each activity to compare them with its boxplot. K-means uses all three standardized sensor magnitudes. The elbow is estimated on 20,000 reproducible samples, followed by fitting on all recordings. An upper Tukey fence on within-cluster distances identifies outliers. DBSCAN is demonstrated on the same sample.
- Outlier injection reaches the requested total density relative to the original mean and standard deviation. The handout's `mean + sign * k * sd + q` formula can produce an inlier for a negative sign. The script repeats such proposals until they exceed the original limits. This literal interpretation can favour upper-tail outliers. Recomputing the limits after injection can give a different density.
- Linear prediction uses subjects 0–9 for training, 10–11 to select the best tested order, and 12–14 to measure prediction error. Training takes one window every 20 readings. Only injected readings are replaced, in time order.
- The activity comparison uses a permutation test on participant-level means, shuffling activity labels within each participant. This preserves the repeated-measures structure. It tests an overall difference between activities.
- Feature extraction follows the temporal and spectral interpretation of exercise 4.2. It uses 102-sample windows (approximately two seconds at 51.2 Hz), 50% overlap, and restarts at each activity change. Each device contributes separate feature rows. The 101 features include per-axis statistics, pairwise correlations, spectral descriptors and averaged acceleration/rotation energies. Direction-dependent biomechanical descriptors are outside this interpretation. The source is Zhang and Sawchuk (2011), *A Feature Selection-Based Framework for Human Activity Recognition Using Wearable Multimodal Sensors*, DOI 10.4108/icst.bodynets.2011.247018.
- ReliefF uses 256 reproducible query windows and searches for neighbours among all extracted windows. PCA and both feature rankings show the same example window.

"""FORTH-TRACE laboratory assignment, 
Part A. Python, NumPy, and SciPy scripts and functions 
for data preparation and feature engineering"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# The dataset folder is next to this script on every computer.
DATA_PATH = Path(__file__).resolve().parent / "FORTH_TRACE_DATASET-master"


# A.2: Each row keeps the twelve columns specified in the dataset README.
def load_subject(subject):
    folder = DATA_PATH / f"part{subject}"
    devices = []
    for device in range(1, 6):
        file = folder / f"part{subject}dev{device}.csv"
        table = pd.read_csv(file, header=None)
        devices.append(table.to_numpy())
    return np.vstack(devices)


def magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


# A.3.1-3.5: Use the same activity groups for both univariate methods.
def z_outliers(values, k):
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    lower = mean - k * sd
    upper = mean + k * sd
    return (values < lower) | (values > upper)


def acceleration_outliers(acc, labels):
    activities = np.unique(labels)
    groups = [acc[labels == a] for a in activities]
    fig, ax = plt.subplots()
    ax.boxplot(groups, tick_labels=activities, flierprops={"markersize": 1})
    ax.set(xlabel="Activity", ylabel="Acceleration magnitude", title="A.3.1  All subjects and devices")

    thresholds = [3, 3.5, 4]
    masks = np.zeros((3, len(acc)), dtype=bool)
    rows = []
    for activity, group in zip(activities, groups):
        q1, q3 = np.percentile(group, [25, 75])
        limit = 1.5 * (q3 - q1)
        box = (group < q1 - limit) | (group > q3 + limit)
        row = [activity, len(group), 100 * box.sum() / len(group)]
        for j, k in enumerate(thresholds):
            out = z_outliers(group, k)
            masks[j, labels == activity] = out
            row.append(100 * out.sum() / len(group))
        rows.append(row)

    table = pd.DataFrame(rows, columns=["activity", "samples", "box %", "Z3 %", "Z3.5 %", "Z4 %"])
    print(table.round(2).to_string(index=False))
    print("Highest boxplot density: activity", table.loc[table["box %"].idxmax(), "activity"])
    print("Lowest boxplot density: activity", table.loc[table["box %"].idxmin(), "activity"])
    print("Boxplots use quartiles. Extreme values can widen the Z-score limits by increasing the SD.")

    # Draw the whole signal in blue, then mark the outliers in red.
    fig, axes = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(10, 7), layout="constrained")
    for ax, k, mask in zip(axes, thresholds, masks):
        ax.plot(acc, ".", color="blue", markersize=1)
        indices = np.flatnonzero(mask)
        ax.plot(indices, acc[indices], ".", color="red", markersize=2)
        ax.set(title=f"k = {k}   |   blue: ordinary, red: outlier", ylabel="Acc. magnitude")
    axes[-1].set_xlabel("Sample index across subjects and devices")
    return masks


# A.3.6: Lloyd's K-means minimizes the squared distances to the cluster centres.
def k_means(points, n_clusters):
    model = KMeans(n_clusters=n_clusters, n_init=5, random_state=42)
    model.fit(points)
    return model


def plot_clusters(ax, points, out, title):
    for flag, color in [(False, "blue"), (True, "red")]:
        group = points[out == flag]
        ax.scatter(group[:, 0], group[:, 1], group[:, 2], s=2, color=color)
    ax.set(xlabel="Acceleration", ylabel="Gyroscope", zlabel="Magnetometer", title=title)


# A.3.7: Standardization gives each of the three magnitudes comparable weight.
def cluster_outliers(magnitudes, masks):
    scaled = StandardScaler().fit_transform(magnitudes)
    rng = np.random.default_rng(42)
    sample = rng.choice(len(scaled), 20000, replace=False)
    ks = np.arange(1, 11)
    inertia = []
    for k in ks:
        model = k_means(scaled[sample], k)
        inertia.append(model.inertia_)
    # Estimate the elbow by the largest gap below the chord joining the endpoints.
    chord = np.linspace(inertia[0], inertia[-1], len(ks))
    best = ks[np.argmax(chord - inertia)]
    fig, ax = plt.subplots()
    ax.plot(ks, inertia, "o-")
    ax.axvline(best, color="red", linestyle="--")
    ax.set(xlabel="Number of clusters", ylabel="Within-cluster sum of squares", title="A.3.7  Elbow")

    # Fit the selected model on all samples. Use an upper Tukey fence per cluster.
    model = k_means(scaled, best)
    out = np.zeros(len(scaled), dtype=bool)
    for c, centre in enumerate(model.cluster_centers_):
        idx = np.flatnonzero(model.labels_ == c)
        distance = np.linalg.norm(scaled[idx] - centre, axis=1)
        q1, q3 = np.percentile(distance, [25, 75])
        out[idx] = distance > q3 + 1.5 * (q3 - q1)
    print(f"\nA.3.7  Elbow k={best}, K-means outliers={out.sum()} ({100*out.mean():.2f}%)")
    for k, z in zip([3, 3.5, 4], masks):
        print(f"k={k}: both={(out & z).sum()}, K-means only={(out & ~z).sum()}, Z only={(z & ~out).sum()}")
    print("K-means can also flag unusual gyro/magnetic readings at ordinary acceleration.")
    # Compare the methods on the same 3D scatter plot sample.
    fig = plt.figure(figsize=(13, 4), layout="constrained")
    points = magnitudes[sample]
    plot_clusters(fig.add_subplot(131, projection="3d"), points, out[sample], "K-means")
    plot_clusters(fig.add_subplot(132, projection="3d"), points, masks[0, sample], "Z-score (k=3)")

    # A.3.7.1 (bonus): DBSCAN marks isolated samples with label -1.
    noise = DBSCAN(eps=0.25, min_samples=10).fit_predict(scaled[sample]) == -1
    plot_clusters(fig.add_subplot(133, projection="3d"), points, noise, "DBSCAN")
    print("Outliers on the plotted sample (DBSCAN, K-means, Z3):",
          noise.sum(), out[sample].sum(), masks[0, sample].sum())


# A.3.8: Use the original mean and sample SD as fixed reference limits.
def inject_outliers(values, x, k, z, rng):
    clean = np.asarray(values, dtype=float)
    if len(clean) < 2 or not 0 <= x <= 100 or k <= 0 or z <= 0:
        raise ValueError("Use at least two samples, 0 <= x <= 100, k > 0 and z > 0.")
    changed = clean.copy()
    old = z_outliers(clean, k)
    # Round up relative to ALL samples so the requested final density is reached.
    needed = max(0, int(np.ceil(len(clean) * x / 100)) - old.sum())
    targets = rng.choice(np.flatnonzero(~old), needed, replace=False)
    injected = np.zeros(len(clean), dtype=bool)
    injected[targets] = True
    mu, span = clean.mean(), k * clean.std(ddof=1)
    # A negative sign plus positive q can land inside the limits. Redraw such proposals.
    for i in targets:
        while True:
            sign = rng.choice([-1, 1])
            q = rng.uniform(0, z)
            candidate = mu + sign * span + q
            if abs(candidate - mu) > span:
                changed[i] = candidate
                break
    return changed, injected


# A.3.9: Add an intercept column and solve ordinary least squares.
def linear_model(inputs, outputs):
    design = np.column_stack((np.ones(len(inputs)), inputs))
    return np.linalg.lstsq(design, outputs, rcond=None)[0]


# A.3.10: Use past readings only. Consecutive outliers use the earlier repairs.
def repair(trace, noisy, indices, beta):
    p = len(beta) - 1
    repaired = noisy.copy()
    for t in indices:
        repaired[t] = beta[0] + np.dot(beta[1:], repaired[t-p:t])
    return repaired, repaired[indices] - trace[indices]


def prediction_study(traces):
    orders = [1, 2, 4, 8, 16, 32]
    longest = max(orders)
    rng = np.random.default_rng(42)
    training, validation, test = [], [], []
    for number, trace in enumerate(traces):
        # Each subject has five recordings. Keep whole subjects in the same set.
        subject = number // 5
        if subject < 10:
            # Each row holds previous readings and the next true value.
            for t in range(longest, len(trace), 20):
                training.append(trace[t-longest:t+1])
        else:
            # Keep the initial history clean and inject once for all candidate orders.
            noisy = trace.copy()
            tail = trace[longest:]
            noisy[longest:], injected = inject_outliers(tail, 5, 3, tail.std(), rng)
            indices = np.flatnonzero(injected) + longest
            if subject < 12:
                validation.append((trace, noisy, indices))
            else:
                test.append((trace, noisy, indices))

    training = np.array(training)
    past = training[:, :-1]
    actual = training[:, -1]
    best_rmse = np.inf
    for p in orders:
        beta = linear_model(past[:, -p:], actual)
        errors = []
        for trace, noisy, indices in validation:
            repaired, error = repair(trace, noisy, indices, beta)
            errors.extend(error)
        rmse = np.sqrt(np.mean(np.square(errors)))
        print(f"A.3.10  p={p}, validation RMSE={rmse:.4f}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_beta = beta

    # Measure the selected model on separate subjects and plot one example.
    errors = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for number, (trace, noisy, indices) in enumerate(test):
        repaired, error = repair(trace, noisy, indices, best_beta)
        errors.extend(error)
        if number == 0:
            points = indices[indices < 400]
            axes[1].plot(trace[:400], label="Actual")
            axes[1].scatter(points, repaired[points], color="red", label="Predicted")
    print(f"Best tested p={len(best_beta)-1}, test RMSE={np.sqrt(np.mean(np.square(errors))):.4f}")
    axes[0].hist(errors, bins=60, log=True)
    axes[0].set(xlabel="Prediction minus actual", ylabel="Count (log)", title="Test errors")
    axes[1].set(xlabel="Sample, subject 12 / device 1", ylabel="Acceleration magnitude")
    axes[1].legend()


# A.4.1: Participants are the experimental units. Keep the repeated measures paired.
def compare_activity_means(subject_means):
    means = np.asarray(subject_means)
    print("\nA.4.1  Mean acceleration magnitude (equal weight per participant):")
    print(pd.Series(means.mean(axis=0), index=range(1, 17)).round(3).to_string())
    # Under the null of exchangeable activity labels, shuffle within each participant.
    observed = np.var(means.mean(axis=0))
    rng = np.random.default_rng(42)
    exceed = 0
    for _ in range(9999):
        shuffled = rng.permuted(means, axis=1)
        exceed += np.var(shuffled.mean(axis=0)) >= observed
    p_value = (1 + exceed) / 10000
    print(f"Paired permutation test: variance of activity means={observed:.5f}, p={p_value:.4g}.")
    print("Evidence of an overall activity effect." if p_value < .05 else "No significant overall effect.")
    print("The omnibus result does not identify specific pairs of activities.")

# A.4.2: Zhang & Sawchuk (2011), Table 1 and section 3.3
# Each row of a signal matrix is a two-second window of one sensor axis.
def crossing_rate(signal, level):
    above = signal > level
    return np.sum(above[:, 1:] != above[:, :-1], axis=1) / signal.shape[1]


def spectral_features(signal, sampling_rate):
    power = np.abs(np.fft.rfft(signal, axis=1))**2
    frequencies = np.fft.rfftfreq(signal.shape[1], 1 / sampling_rate)
    dominant = frequencies[np.argmax(power, axis=1)]
    entropy = np.nan_to_num(stats.entropy(power, axis=1, base=2))
    # Removing the mean excludes DC. By Parseval, this equals FFT energy divided by N.
    centred = signal - signal.mean(axis=1, keepdims=True)
    energy = np.sum(centred**2, axis=1)
    return entropy, dominant, energy


def axis_features(signal, sampling_rate):
    # Apply each time-domain or spectral statistic to the samples in each row.
    mean = signal.mean(axis=1, keepdims=True)
    entropy, dominant, energy = spectral_features(signal, sampling_rate)
    return {
        "mean": mean.ravel(), "median": np.median(signal, axis=1),
        "sd": signal.std(axis=1), "variance": signal.var(axis=1),
        "rms": np.sqrt(np.mean(signal**2, axis=1)),
        "derivative": np.diff(signal, axis=1).mean(axis=1) * sampling_rate,
        "skew": np.nan_to_num(stats.skew(signal, axis=1)),
        "kurtosis": np.nan_to_num(stats.kurtosis(signal, axis=1)),
        "iqr": stats.iqr(signal, axis=1), "zero_cross": crossing_rate(signal, 0),
        "mean_cross": crossing_rate(signal, mean), "entropy": entropy,
        "dominant_hz": dominant, "energy": energy,
    }


def extract_features(recording, subject):
    # The sensors record 51.2 samples per second. Each window spans two seconds.
    sampling_rate = 51.2
    window_size = round(2 * sampling_rate)
    labels = recording[:, 11].astype(int)
    # Start again at each activity change. Consecutive windows overlap by half.
    changes = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    boundaries = [0] + changes.tolist() + [len(labels)]
    starts = []
    for first, stop in zip(boundaries[:-1], boundaries[1:]):
        starts.extend(range(first, stop - window_size + 1, window_size // 2))

    features = {}
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]
    standardized = {}
    for column, axis in enumerate(axes, start=1):
        signal = np.array([recording[t:t+window_size, column] for t in starts])
        for name, values in axis_features(signal, sampling_rate).items():
            features[f"{axis}_{name}"] = values
        # A constant axis has no variation. Give its correlations a value of zero.
        standardized[axis] = np.nan_to_num(stats.zscore(signal, axis=1))

    # Correlation is the mean product of two standardized axes in the same window.
    for i, first in enumerate(axes):
        for second in axes[i+1:]:
            features[f"corr_{first}_{second}"] = (standardized[first] * standardized[second]).mean(axis=1)
    frame = pd.DataFrame(features)
    frame["AAE"] = frame[["ax_energy", "ay_energy", "az_energy"]].mean(axis=1)
    frame["ARE"] = frame[["gx_energy", "gy_energy", "gz_energy"]].mean(axis=1)

    # Identify each feature vector by the last reading included in the window.
    ends = np.array(starts, dtype=int) + window_size - 1
    frame["subject"] = subject
    frame["device"] = int(recording[0, 0])
    frame["timestamp"] = recording[ends, 10]
    frame["activity"] = labels[ends]
    return frame


# A.4.3-4.4.1: Project Z-normalized features onto orthogonal directions.
def pca_study(features):
    scaled = StandardScaler().fit_transform(features)
    model = PCA(svd_solver="full")
    model.fit(scaled)
    variance = model.explained_variance_ratio_
    cumulative = np.cumsum(variance)
    count = np.searchsorted(cumulative, .75) + 1
    print("\nA.4.4  Variance explained by each eigenvector (%)")
    print(pd.DataFrame({"individual": 100*variance, "cumulative": 100*cumulative},
                       index=np.arange(1, len(variance)+1)).round(3).to_string())
    print(f"{count} components explain {100*cumulative[count-1]:.2f}%.")
    # A score is the dot product of the centered, standardized row and an eigenvector.
    print("First retained PCA scores:", model.transform(scaled[:1])[0, :count].round(3))


# A.4.5: Fisher rewards differences between class means relative to within-class spread.
def fisher_score(values, labels):
    between = np.zeros(values.shape[1])
    within = np.zeros(values.shape[1])
    mean = values.mean(axis=0)
    for label in np.unique(labels):
        group = values[labels == label]
        centre = group.mean(axis=0)
        between += len(group) * (centre - mean)**2
        within += ((group - centre)**2).sum(axis=0)
    scores = np.divide(between, within, out=np.zeros_like(between), where=within > 0)
    scores[(within == 0) & (between > 0)] = np.inf
    return scores


def relief_f(values, labels, neighbors=10, n_queries=256):
    # Scale differences to [0, 1]. Draw reproducible queries; all windows are neighbours.
    x = MinMaxScaler().fit_transform(values)
    rng = np.random.default_rng(42)
    queries = rng.choice(len(x), min(n_queries, len(x)), replace=False)
    weights = np.zeros(x.shape[1])
    for query in queries:
        # Find the distance from this window to every candidate neighbour.
        # Manhattan distance adds the absolute differences across features.
        distances = np.sum(np.abs(x - x[query]), axis=1)
        own_class = labels[query]
        other_probability = np.mean(labels != own_class)
        for activity in np.unique(labels):
            indices = np.flatnonzero(labels == activity)
            probability = len(indices) / len(x)
            indices = indices[indices != query]
            if len(indices) == 0:
                continue
            order = np.argsort(distances[indices])
            closest = indices[order[:neighbors]]
            difference = np.abs(x[closest] - x[query]).mean(axis=0)
            # Close matches should agree. Other classes should differ.
            if activity == own_class:
                weights -= difference
            else:
                weights += difference * probability / other_probability
    return weights / len(queries)


def rank_features(features, labels):
    values = features.to_numpy()
    # A.4.6-4.6.1: Select columns by score and keep their original measured values.
    scores_by_method = {
        "Fisher": fisher_score(values, labels),
        "ReliefF": relief_f(values, labels),
    }
    for name, scores in scores_by_method.items():
        table = pd.DataFrame(index=features.columns)
        table["score"] = scores
        table["example"] = features.iloc[0]
        print(f"\nA.4.6  {name}: top ten and values for the example window")
        print(table.nlargest(10, "score").round(5))


# A.1: Run the exercises in the order used in the handout.
def main():
    traces, magnitudes, labels, subject_means = [], [], [], []
    for subject in range(15):
        data = load_subject(subject)
        activity = data[:, 11].astype(int)
        # Take the Euclidean length of each three-axis sensor vector.
        acc = magnitude(data[:, 1], data[:, 2], data[:, 3])
        gyro = magnitude(data[:, 4], data[:, 5], data[:, 6])
        magnet = magnitude(data[:, 7], data[:, 8], data[:, 9])
        mag = np.column_stack((acc, gyro, magnet))
        magnitudes.append(mag)
        labels.append(activity)
        for device in range(1, 6):
            traces.append(acc[data[:, 0] == device])
        means = []
        for label in range(1, 17):
            means.append(acc[activity == label].mean())
        subject_means.append(means)
    magnitudes = np.vstack(magnitudes)
    labels = np.concatenate(labels)
    print(f"A.2  Loaded {len(labels):,} samples from {len(traces)} recordings.")
    masks = acceleration_outliers(magnitudes[:, 0], labels)
    cluster_outliers(magnitudes, masks)
    prediction_study(traces)
    compare_activity_means(subject_means)

    # Process one subject at a time so raw axis windows do not fill memory.
    frames = []
    for subject in range(15):
        data = load_subject(subject)
        for device in range(1, 6):
            frames.append(extract_features(data[data[:, 0] == device], subject))
    data = pd.concat(frames, ignore_index=True)
    # One row per window: X contains its features and y its activity label.
    metadata = ["subject", "device", "timestamp", "activity"]
    X = data.drop(columns=metadata)
    y = data["activity"].to_numpy()
    print(f"\nA.4.2  {len(X):,} window-end instants, {X.shape[1]} features.")
    print("Example window used for PCA, Fisher and ReliefF:")
    print(data.loc[0, metadata].to_string())
    pca_study(X)
    rank_features(X, y)
    plt.show()


if __name__ == "__main__":
    main()
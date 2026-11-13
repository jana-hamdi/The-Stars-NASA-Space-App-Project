# 🌕 Moon Dataset Analysis — EDA & Study Notes

**Project:** The Stars — NASA Space Apps Challenge  
**Dataset:** Lunar Digital Elevation Model (LDEM)  
**Stage:** Exploratory Data Analysis (EDA)  
**Status:** Completed — Ready for Terrain Feature Extraction

---

## 1. Project Overview

The Stars aims to help researchers identify locations on Earth that share similar terrain characteristics with selected regions on the Moon or Mars.

These terrestrial locations could serve as analog sites for conducting experiments, testing scientific models, and studying planetary surface conditions.

To achieve this, we need to:

1. Understand and inspect planetary elevation datasets.
2. Extract meaningful terrain features.
3. Represent geographical regions using numerical feature vectors.
4. Compare Earth and planetary terrain characteristics.
5. Apply clustering techniques to discover potentially similar regions.

This notebook documents the initial analysis of the lunar elevation dataset.

## 2. Dataset Description

**LDEM (Lunar Digital Elevation Model)** is a digital representation of the Moon's surface, where each raster cell contains a numerical elevation value.

Unlike ordinary images, an LDEM dataset stores numerical measurements rather than RGB color information.

### Dataset File

```text
Data/
└── Moon_Data/
    └── ldem_16_float.img
```

The `.img` file is a binary raster file containing floating-point elevation values.

**Important:** The file's metadata and product documentation should be checked to verify its exact format, units, projection, spatial resolution, and geographic referencing.

## 3. Loading the Dataset

We used NumPy to read the binary raster file and convert it into a two-dimensional array.

```python
import os
import numpy as np
import matplotlib.pyplot as plt

moon_path = os.path.abspath(
    "../Data/Moon_Data/ldem_16_float.img"
)

hgt = np.fromfile(moon_path, dtype=np.float32)
hgt = hgt.reshape(2880, 5760)
```

### Understanding the implementation

- `np.fromfile()`: Reads binary data from the file.
- `dtype=np.float32`: Interprets each value as a 32-bit floating-point number.
- `reshape(2880, 5760)`: Converts the one-dimensional array into a two-dimensional raster.

### Dataset Structure

| Property | Value |
|---|---:|
| Rows | 2,880 |
| Columns | 5,760 |
| Total Pixels | 16,588,800 |
| Data Type | float32 |
| Dimensions | 2D |

The dataset contains approximately **16.6 million elevation values**.

## 4. Basic Statistical Analysis

We calculated descriptive statistics to understand the elevation values and their distribution.

```python
print("Minimum:", hgt.min())
print("Maximum:", hgt.max())
print("Mean:", np.mean(hgt))
print("Median:", np.median(hgt))
print("Standard Deviation:", np.std(hgt))
```

### Results

| Statistic | Value |
|---|---:|
| Minimum | -8.981698 km |
| Maximum | 10.6857395 km |
| Mean | -0.51928246 km |
| Median | -0.79440415 km |
| Standard Deviation | 2.2040894 km |

### Statistical Interpretation

**Minimum and Maximum**

Describe the smallest and largest elevation values in the dataset.

**Mean**

The arithmetic average of all elevation values.

**Median**

The middle value after sorting the observations. It is less sensitive to extreme values than the mean.

**Standard Deviation**

Measures how spread out the elevation values are around the mean.

We used `np.std()`, which calculates population standard deviation by default (`ddof=0`).

### Initial Observation

The mean is greater than the median:

```text
Mean   = -0.519 km
Median = -0.794 km
```

This may suggest right-skewness, but the distribution should be inspected before reaching a definitive conclusion.

## 5. Global Lunar Elevation Map

We visualized the raster using Matplotlib.

```python
plt.figure(figsize=(16, 8))

plt.imshow(
    hgt,
    cmap="terrain",
    vmin=-9,
    vmax=11
)

plt.colorbar(label="Elevation (km)")
plt.title("Global Lunar Elevation Map")
plt.xlabel("Pixel Column")
plt.ylabel("Pixel Row")

plt.show()
```

### Visualization Concepts

- `imshow()`: Displays the two-dimensional array as an image.
- `cmap="terrain"`: Applies a terrain-oriented color map.
- `vmin` and `vmax`: Define the elevation range used for color mapping.
- `colorbar()`: Displays the relationship between colors and elevation values.

**Important:** The axes currently represent pixel rows and columns, not actual latitude and longitude. Geographic coordinates require the appropriate projection and metadata.

## 6. Elevation Distribution — Histogram

A histogram helps us understand how frequently different elevation values occur.

```python
plt.figure(figsize=(12, 6))

plt.hist(
    hgt.flatten(),
    bins=100,
    color="steelblue",
    edgecolor="black",
    alpha=0.7
)

plt.axvline(
    hgt.mean(),
    color="red",
    linestyle="--",
    label=f"Mean: {hgt.mean():.2f} km"
)

plt.axvline(
    np.median(hgt),
    color="orange",
    linestyle="--",
    label=f"Median: {np.median(hgt):.2f} km"
)

plt.title("Distribution of Lunar Elevation")
plt.xlabel("Elevation (km)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.2)

plt.show()
```

### Why `flatten()`?

The original dataset is a two-dimensional array.

`flatten()` converts it into a one-dimensional array so that we can analyze the overall distribution without considering spatial positions.

### Why 100 bins?

The elevation range is divided into 100 intervals. Each bar represents the frequency of observations within its interval.

The histogram plot still needs visual review before making final observations about the distribution's shape.

## 7. Percentile Analysis

Percentiles describe the position of values within a sorted dataset.

For example, the 90th percentile represents a value at or below which approximately 90% of observations fall.

```python
percentiles = [0, 10, 25, 50, 75, 90, 95, 99, 100]

for p in percentiles:
    print(
        f"{p}th Percentile:",
        np.percentile(hgt, p)
    )
```

### Results

| Percentile | Elevation (km) |
|---|---:|
| P0 | -8.981698 |
| P10 | -2.8624027 |
| P25 | -1.9044077 |
| P50 | -0.79440415 |
| P75 | 0.62809545 |
| P90 | 2.488769 |
| P95 | 3.5860682 |
| P99 | 5.6925354 |
| P100 | 10.6857395 |

### Interpretation

- 50% of observations are at or below the median.
- 90% of observations are at or below approximately 2.489 km.
- The highest 1% begins around 5.693 km.
- Percentiles provide more information about the distribution than minimum and maximum values alone.

## 8. Quartiles and Interquartile Range (IQR)

Quartiles divide sorted observations into four equal parts.

- **Q1:** 25th percentile.
- **Q2:** 50th percentile (Median).
- **Q3:** 75th percentile.

The Interquartile Range measures the spread of the middle 50% of the data.

\[
IQR = Q3 - Q1
\]

### Implementation

```python
Q1 = np.percentile(hgt, 25)
Q3 = np.percentile(hgt, 75)

IQR = Q3 - Q1

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
```

### Results

```text
Q1:  -1.9044077
Q3:   0.62809545
IQR:  2.5325031
```

Therefore:

\[
IQR = 0.62809545 - (-1.9044077)
\]

\[
\boxed{IQR = 2.5325031\ km}
\]

### Why is IQR useful?

IQR measures the spread of the central portion of the data and is less sensitive to extreme values than the full range.

It can also help identify potential statistical outliers using:

\[
Lower\ Fence = Q1 - 1.5(IQR)
\]

\[
Upper\ Fence = Q3 + 1.5(IQR)
\]

Values outside these boundaries are potential statistical outliers.

However, an extreme elevation value may represent a genuine lunar feature. It should not automatically be treated as an error or removed.

## 9. P1 and P99 — Extreme Percentiles

We calculated the 1st and 99th percentiles to understand the tails of the distribution.

```python
P1 = np.percentile(hgt, 1)
P99 = np.percentile(hgt, 99)

print("P1:", P1)
print("P99:", P99)
```

### Results

```text
P1:  -5.3264875
P99:  5.6925354
```

Approximately 98% of the observations fall between P1 and P99.

These values can help us choose visualization limits and understand extreme observations without relying entirely on the minimum and maximum.

They do not imply that observations outside this range should be deleted.

## 10. Complete Statistical Summary

| Metric | Result |
|---|---:|
| Rows | 2,880 |
| Columns | 5,760 |
| Total Pixels | 16,588,800 |
| Data Type | float32 |
| Minimum | -8.981698 km |
| Maximum | 10.6857395 km |
| Mean | -0.51928246 km |
| Median | -0.79440415 km |
| Standard Deviation | 2.2040894 km |
| Q1 | -1.9044077 km |
| Q3 | 0.62809545 km |
| IQR | 2.5325031 km |
| P1 | -5.3264875 km |
| P99 | 5.6925354 km |

## 11. Progress Tracker

| Task | Status |
|---|---|
| Inspect Moon dataset files | Completed |
| Load the LDEM file | Completed |
| Convert data into NumPy array | Completed |
| Inspect array dimensions and data type | Completed |
| Calculate descriptive statistics | Completed |
| Visualize global elevation | Completed |
| Prepare histogram | Pending visual review |
| Calculate percentiles | Completed |
| Calculate quartiles and IQR | Completed |
| Terrain Feature Extraction | Not Started |
| Earth–Moon Comparative Analysis | Not Started |
| Clustering | Not Started |

## 12. Next Stage — Terrain Feature Extraction

Elevation alone is not enough to describe terrain morphology.

Our next step is to extract additional terrain features.

### 12.1 Slope

Measures how steep the terrain is.

It helps distinguish flat surfaces from steep slopes.

### 12.2 Surface Roughness

Measures local surface irregularity.

The calculation depends on the selected neighborhood size and roughness definition.

### 12.3 Local Relief

Measures elevation variation within a defined neighborhood.

For example:

\[
Local\ Relief = Local\ Maximum - Local\ Minimum
\]

### 12.4 Curvature

Describes how the terrain surface bends.

It can help distinguish peaks, basins, concave regions, and convex regions.

### Planned Workflow

1. Review the LDEM metadata and verify its spatial reference.
2. Calculate terrain features using appropriate methods.
3. Inspect and visualize the extracted features.
4. Repeat compatible feature extraction for Earth elevation data.
5. Harmonize units, resolutions, and spatial scales.
6. Build feature vectors for geographical regions.
7. Apply clustering to discover potentially similar terrain patterns.

**Scientific consideration:** Earth and Moon datasets must be spatially and methodologically comparable before clustering. Differences in projection, resolution, measurement units, and feature calculation can significantly affect the results.

## 13. Self-Review Questions

Before moving to feature extraction, make sure you can answer:

1. What is the difference between array shape and total pixel count?
2. Why did we specify `dtype=np.float32`?
3. What is the difference between mean and median?
4. What does standard deviation measure?
5. What is the difference between an elevation map and a histogram?
6. Why did we use `flatten()`?
7. What does the 90th percentile represent?
8. How are Q1, Q3, and IQR related?
9. Why are pixel rows and columns not equivalent to latitude and longitude?
10. Why is elevation alone insufficient for identifying similar terrain?

---

## Conclusion

In this stage, we completed the initial Exploratory Data Analysis of the lunar elevation dataset.

We loaded the raster, inspected its structure, calculated descriptive statistics, explored percentiles and quartiles, and prepared visualizations to understand the elevation distribution.

No Machine Learning or clustering has been performed yet.

The purpose of this stage is to understand the raw data and prepare a reliable foundation for terrain feature extraction and subsequent Earth–Moon comparisons.

**Next Milestone: Lunar Metadata Verification & Terrain Feature Extraction.**
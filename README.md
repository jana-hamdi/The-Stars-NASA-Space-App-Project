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

### Metadata Consistency Check

The downloaded LDEM_16_FLOAT.IMG file was
validated against the official PDS archive.

- Product ID: LDEM_16_FLOAT
- Product Version: V3.1
- Actual file size: 66,355,200 bytes
- Expected file size: 66,355,200 bytes
- Array dimensions: 2880 × 5760
- Data type: float32 (Little Endian)

A discrepancy was observed between the MINIMUM
and MAXIMUM values reported in the official PDS
label and the statistics calculated from the
downloaded binary array.

| Statistic | PDS Label | Actual Array |
|-----------|-----------|--------------|
| Minimum | -9.075 km | -8.981698 km |
| Maximum | 10.790 km | 10.6857395 km |

The discrepancy is documented and the original
pixel values are preserved without modification.


### Slope Threshold Analysis

To better understand the distribution of lunar terrain slopes, we calculated the percentage of pixels exceeding several slope thresholds.

| Slope Threshold | Percentage of Pixels |
|---|---:|
| Greater than 5° | 31.6946% |
| Greater than 10° | 10.9330% |
| Greater than 15° | 3.2954% |
| Greater than 20° | 0.8097% |
| Greater than 25° | 0.1294% |
| Greater than 30° | 0.0041% |

**Key Observations:**

- Approximately 68.3% of the pixels have slopes of 5° or less.
- Approximately 89.1% have slopes of 10° or less.
- Only 0.81% of the pixels have slopes greater than 20°.
- Slopes exceeding 30° are extremely rare, representing only 0.0041% of the dataset.

These results indicate that most of the analyzed lunar surface has relatively low slope values, while steep terrain occupies a very small proportion of the dataset.

The threshold percentages can later be used as terrain features when comparing lunar regions with potential terrestrial analog sites.

### Slope Outlier Analysis

We applied the Interquartile Range (IQR) method to identify unusually high slope values.

**Results:**

| Metric | Value |
|---|---:|
| Q1 | 0.9899° |
| Q3 | 6.1257° |
| IQR | 5.1358° |
| Lower Bound | -6.7138° |
| Upper Bound | 13.8295° |
| Number of Outliers | 736,387 |
| Outlier Percentage | 4.4391% |

**Interpretation:**

Approximately 4.44% of the lunar pixels have slope values above the upper IQR threshold of 13.83°.

These values are not necessarily data errors. They may represent genuine steep geological features, such as crater walls, crater rims, and mountain slopes.

**Decision:**

No outliers were removed. All original slope values were preserved because steep terrain is an important characteristic for identifying potential lunar analog sites on Earth.


### Lunar Terrain Roughness Analysis

We calculated the **Terrain Ruggedness Index (TRI)** using a 3 × 3 moving window.

For each central pixel, TRI measures the combined elevation differences between that pixel and its eight neighboring pixels.

The formula used:

\[
TRI = \sqrt{\sum_{i=1}^{8}(z_i-z_c)^2}
\]

Where:
- \(z_c\) is the central pixel elevation.
- \(z_i\) represents the elevation of each neighboring pixel.

#### Statistical Results

| Metric | Value |
|---|---:|
| Shape | (2880, 5760) |
| Minimum | 0.000439 km |
| Maximum | 3.658554 km |
| Mean | 0.389308 km |
| Median | 0.283364 km |

#### Key Observations

- The mean roughness is higher than the median, suggesting a distribution with a higher-value tail.
- Low TRI values indicate relatively small elevation differences within the local neighborhood.
- High TRI values indicate greater local terrain variability.
- Circular patterns in the roughness map may correspond to crater-related terrain structures.

#### Methodological Notes

The roughness values were calculated using a 3 × 3 neighborhood and the Riley TRI method. The resulting values are expressed in kilometers because the input elevation data is measured in kilometers.

The selected window size determines the spatial scale of the roughness measurement. These values describe terrain variability at the DEM resolution and should not be interpreted as direct measurements of individual rocks or small surface objects.

#### Roughness Percentile Analysis

To understand the distribution of lunar terrain roughness, we calculated the following percentiles:

| Percentile | Roughness (km) |
|---|---:|
| P0 | 0.0004 |
| P10 | 0.0381 |
| P25 | 0.1125 |
| P50 | 0.2834 |
| P75 | 0.5528 |
| P90 | 0.8956 |
| P95 | 1.1341 |
| P99 | 1.6296 |
| P100 | 3.6586 |

**Key Observations:**

- 50% of the pixels have roughness values below or equal to 0.2834 km.
- 75% have values below or equal to 0.5528 km.
- 90% have values below or equal to 0.8956 km.
- The highest 1% of pixels have roughness values above approximately 1.6296 km.

These results show that most of the analyzed lunar surface has relatively low local elevation variability, while a smaller portion exhibits substantially higher roughness.

No values were removed based on percentile analysis.

#### Local Relief Percentile Analysis

We calculated percentiles to understand the distribution of local elevation differences across the lunar surface.

| Percentile | Local Relief (km) |
|---|---:|
| P0 | 0.0004 |
| P10 | 0.0356 |
| P25 | 0.1014 |
| P50 | 0.2584 |
| P75 | 0.5146 |
| P90 | 0.8469 |
| P95 | 1.0804 |
| P99 | 1.5646 |
| P100 | 3.5431 |

**Key Observations:**

- 50% of the pixels have local relief values below or equal to 0.2584 km.
- 75% have values below or equal to 0.5146 km.
- 90% have values below or equal to 0.8469 km.
- The highest 1% of pixels have local relief values above approximately 1.5646 km.

These results indicate that most local neighborhoods have relatively small elevation ranges, while a smaller portion exhibits much larger vertical differences.

No values were removed based on percentile analysis.

## 4. Curvature Extraction

### What is Curvature?

Curvature describes how the lunar surface bends or changes shape.

Unlike Slope, which measures how steep the surface is, Curvature measures how the slope changes across the surface.

It helps distinguish between:
- **Convex surfaces:** Areas that bend outward, such as hilltops.
- **Concave surfaces:** Areas that bend inward, such as crater floors.
- **Nearly flat surfaces:** Areas with curvature values close to zero.

### Calculation Method

We calculated **Mean Curvature** using the second derivatives of the elevation data.

First, we calculated the first derivatives:

```python
dz_dy, dz_dx = np.gradient(
    moon_clean,
    pixel_size,
    pixel_size
)
```

Then, we calculated the second derivatives:

```python
d2z_dyy, d2z_dyx = np.gradient(
    dz_dy,
    pixel_size,
    pixel_size
)

d2z_dxy, d2z_dxx = np.gradient(
    dz_dx,
    pixel_size,
    pixel_size
)
```

Finally, we applied the Mean Curvature formula:

\[
H = \frac{
(1+z_y^2)z_{xx}
-2z_xz_yz_{xy}
+(1+z_x^2)z_{yy}
}{
2(1+z_x^2+z_y^2)^{3/2}
}
\]

Where:
- \(z_x\) and \(z_y\) are the first derivatives of elevation.
- \(z_{xx}\) and \(z_{yy}\) are the second derivatives.
- \(z_{xy}\) is the mixed second derivative.

The resulting curvature values are measured in **km⁻¹**.

### Results

| Metric | Value |
|---|---:|
| Shape | (2880, 5760) |
| Minimum Curvature | -0.17698418 km⁻¹ |
| Maximum Curvature | 0.20014636 km⁻¹ |
| Mean Curvature | 0.0000234068 km⁻¹ |
| Median Curvature | 0.00041157304 km⁻¹ |

### Understanding the Results

- **Negative Curvature:** Under the sign convention used in our formula, negative values can indicate convex surface shapes, such as hilltops.
- **Positive Curvature:** Positive values can indicate concave surface shapes, such as bowl-shaped depressions.
- **Values Near Zero:** Indicate areas with relatively weak curvature.

The mean is very close to zero, suggesting that positive and negative curvature values largely balance each other when averaged across the entire dataset.

### Visualization

The curvature map uses a diverging color scale to distinguish between negative and positive curvature values:

- Negative values represent one direction of surface bending.
- Positive values represent the opposite direction.
- Values close to zero represent weak curvature.

Circular patterns may correspond to crater-related terrain structures, while irregular patterns may reflect variations in the lunar surface shape. These interpretations require further geographic and geological validation.

### Important Notes and Limitations

1. The calculation uses a constant pixel size of 1.89521 km in both directions.
2. The dataset uses a Simple Cylindrical projection, where actual east-west ground spacing varies with latitude.
3. Curvature is sensitive to elevation noise and spatial resolution.
4. Numerical derivatives at image boundaries may introduce less reliable values.
5. The dataset metadata mentions possible artifacts at 45-degree latitude band boundaries. These should be investigated before treating the results as final.

Therefore, the current Mean Curvature map is a **preliminary terrain feature extraction**, not a fully corrected geospatial curvature product.

---

## Summary: Lunar Terrain Features

We have extracted four terrain features from the lunar elevation dataset:

| Feature | What It Measures |
|---|---|
| Slope | Surface steepness |
| Roughness | Local elevation variability relative to the center pixel |
| Local Relief | Difference between maximum and minimum elevation in a neighborhood |
| Curvature | Surface bending and shape |

Elevation is the original input dataset used to derive these features.

These features will form the basis of our next stage: dividing the terrain into spatial regions, generating regional feature vectors, and eventually comparing and clustering lunar, Martian, and terrestrial landscapes.
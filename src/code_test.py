import numpy as np
from sklearn.cluster import DBSCAN
from collections import Counter

# Load the data file where each row is: class_id x_c y_c w h
data = np.loadtxt("/home/hunglt31/examark/dataset1/metadata/valid/labels/page_001_exam_id_v0_png.rf.513610f9ef5c64bd6103282e9a4ef83b_aug.txt")

# Extract columns: class_ids, x_center, y_center
class_ids = data[:, 0].astype(int)
x_centers = data[:, 1].reshape(-1, 1)
y_centers = data[:, 2].reshape(-1, 1)

# Cluster x_center values.
# Adjust eps so that DBSCAN produces (ideally) 3 clusters for x_center.
db_x = DBSCAN(eps=0.05, min_samples=1).fit(x_centers)
x_labels = db_x.labels_

# Cluster y_center values.
# Adjust eps so that DBSCAN produces (ideally) 10 clusters for y_center.
db_y = DBSCAN(eps=0.05, min_samples=1).fit(y_centers)
y_labels = db_y.labels_

# Order the x clusters by their mean x_center value.
unique_x = np.unique(x_labels)
x_cluster_means = []
for label in unique_x:
    idx = np.where(x_labels == label)[0]
    mean_val = np.mean(x_centers[idx])
    x_cluster_means.append((label, mean_val))
x_cluster_means.sort(key=lambda x: x[1])
ordered_x_labels = [t[0] for t in x_cluster_means]

# Order the y clusters by their mean y_center value.
unique_y = np.unique(y_labels)
y_cluster_means = []
for label in unique_y:
    idx = np.where(y_labels == label)[0]
    mean_val = np.mean(y_centers[idx])
    y_cluster_means.append((label, mean_val))
y_cluster_means.sort(key=lambda x: x[1])
ordered_y_labels = [t[0] for t in y_cluster_means]

# Number of clusters determines the matrix shape.
n_cols = len(ordered_x_labels)   # expected to be 3
n_rows = len(ordered_y_labels)   # expected to be 10
print("Detected clusters -> rows (y):", n_rows, "columns (x):", n_cols)

# Create an empty matrix filled with -1 (as a placeholder).
matrix = np.full((n_rows, n_cols), -1, dtype=int)

# Create a dictionary to accumulate class_ids for each cell:
# key: (row_index, col_index) determined from ordered clusters.
cell_dict = {}
for i in range(len(data)):
    # Get the cluster labels for this detection.
    x_lab = x_labels[i]
    y_lab = y_labels[i]
    # Find the ordered index (i.e. column and row in the final matrix).
    try:
        col_index = ordered_x_labels.index(x_lab)
        row_index = ordered_y_labels.index(y_lab)
    except ValueError:
        continue
    cell_dict.setdefault((row_index, col_index), []).append(class_ids[i])

# For each cell, choose the majority class_id if available.
for (r, c), ids in cell_dict.items():
    majority = Counter(ids).most_common(1)[0][0]
    matrix[r, c] = majority

print("Final matrix (rows: y_center clusters, columns: x_center clusters):")
print(matrix)
print("Matrix shape:", matrix.shape)

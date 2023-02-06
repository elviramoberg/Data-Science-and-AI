import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


# ------------------------------------------1A-----------------------------------------------
protein_data = pd.read_csv("data_assignment_3.csv")
plt.scatter(protein_data["phi"], protein_data["psi"], color="salmon", edgecolors="black", s=30)

plt.title("Distribution of Phi and Psi Combinations")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()

# ------------------------------------------1B-----------------------------------------------
histogram, x_edges, y_edges = np.histogram2d(protein_data["phi"], protein_data["psi"], bins=50)
plt.pcolormesh(x_edges, y_edges, histogram.T, cmap="Pastel1")

plt.colorbar()
plt.title("Distribution of Phi and Psi Combinations")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()


# ------------------------------------------2A-----------------------------------------------
kmeans = KMeans(n_clusters=4)
kmeans.fit(protein_data[["phi", "psi"]].values)

# predict the clusters to which each data point belongs
clusters = kmeans.fit_predict(protein_data[["phi", "psi"]].values)

# plot the data points and color them based on the predicted cluster
plt.scatter(protein_data["phi"], protein_data["psi"], c=clusters, cmap="YlOrRd", edgecolors="black")
plt.title("K-means clustered Phi and Psi Combinations")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()

# ------------------------------------------3A-----------------------------------------------

dbscan = DBSCAN(eps=10, min_samples=6)
dbscan.fit(protein_data[["phi", "psi"]].values)

# predict the clusters to which each data point belongs
clusters = dbscan.fit_predict(protein_data[["phi", "psi"]].values)

# plot the data points and color them based on the prediction
plt.scatter(protein_data["phi"], protein_data["psi"], c=clusters, cmap="Accent", edgecolors="black")

plt.title("DBSCAN clustered Phi and Psi Combinations")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()

# ------------------------------------------3B-----------------------------------------------
# set different colors for clusters and outliers
colormarkers = ["darkturquoise", "salmon", "mediumslateblue", "plum", "springgreen", "grey", "navajowhite", "tomato", "tan", "snow", "orchid", "cornflowerblue", "lightgreen"]

# loop through the clusters
for cluster, color in zip(np.unique(clusters), colormarkers):
    # plot each cluster with a different colors
    plt.scatter(protein_data[clusters == cluster]["phi"], protein_data[clusters == cluster]["psi"], c=color, edgecolors="black", s=30)

# plot the outliers with a red x
plt.scatter(protein_data.loc[clusters == -1, "phi"], protein_data.loc[clusters == -1, "psi"], marker="x", c="red", edgecolors="black", s=50)

plt.title("DBSCAN clustered Phi and Psi Combinations with Outliers")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()

# ------------------------------------------3C-----------------------------------------------
outliers = np.sum(clusters == -1)
print("Number of outliers:", outliers)

# count how many times each residue name is an outlier
outlier_residue_name, outlier_count = np.unique(protein_data.loc[clusters == -1, "residue name"], return_counts=True)

# plot a bar chart to show the count of each type of outlier
plt.bar(outlier_residue_name, outlier_count, color="deeppink")
plt.title("Outlier Counts by Residue Type")
plt.xlabel("Residue type")
plt.ylabel("Count")
plt.show()



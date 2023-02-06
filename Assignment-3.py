import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

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

# ------------------------------------------3A-----------------------------------------------

dbscan = DBSCAN(eps=10, min_samples=6)
dbscan.fit(protein_data[["phi", "psi"]].values)

# predict the clusters to which each data point belongs
clusters = dbscan.fit_predict(protein_data[["phi", "psi"]].values)

# plot the data points and color them based on the predicted cluster
plt.scatter(protein_data["phi"], protein_data["psi"], c=clusters, cmap="Pastel2", edgecolors="black")

plt.title("DBSCAN clustered Phi and Psi Combinations")
plt.xlabel("Phi")
plt.ylabel("Psi")
plt.show()


import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt

class HierarchicalClustering:
    def __init__(self, n_clusters=2, linkage='complete'):
        """
        Initialize the hierarchical clustering model.

        Parameters:
        - n_clusters: The desired number of clusters to form
        - linkage: The linkage criterion to use ('single', 'complete', 'average', etc.)
        """
        self.n_clusters = n_clusters              # Number of final clusters
        self.linkage_method = linkage             # Linkage strategy (e.g., 'complete')
        self.cluster_labels = None                # Cluster assignments after fit
        self.linkage_matrix = None                # Linkage matrix from hierarchical clustering

    def fit(self, data_points):
        """
        Fit the hierarchical clustering model to the input data.

        Parameters:
        - data_points: Numpy array of shape (n_samples, n_features)
        """
        # Compute pairwise distances (condensed form)
        distance_matrix = pdist(data_points)

        # Perform hierarchical/agglomerative clustering
        self.linkage_matrix = linkage(distance_matrix, method=self.linkage_method)

        # Assign cluster labels based on linkage matrix and cutoff
        self.cluster_labels = fcluster(self.linkage_matrix, t=self.n_clusters, criterion='maxclust') - 1

    def plot_dendrogram(self):
        """
        Plot the dendrogram based on the linkage matrix.
        """
        if self.linkage_matrix is None:
            raise ValueError("The model must be fitted before plotting the dendrogram.")

        # Create the dendrogram visualization
        plt.figure(figsize=(10, 6))
        dendrogram(self.linkage_matrix)
        plt.title(f'Hierarchical Clustering Dendrogram ({self.linkage_method.capitalize()} Linkage)')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.show()


# -------------------------- Example Usage -----------------------------

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic 2D cluster data from 3 different centers
cluster1 = np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2))
cluster2 = np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2))
cluster3 = np.random.normal(loc=[0, 5], scale=0.5, size=(50, 2))

# Combine all clusters into one dataset
synthetic_data = np.concatenate([cluster1, cluster2, cluster3])

# Create and fit the hierarchical clustering model
clustering_model = HierarchicalClustering(n_clusters=3, linkage='complete')
clustering_model.fit(synthetic_data)

# -------------------------- Plotting -----------------------------

# Plot the cluster assignments
plt.figure(figsize=(12, 5))

# Plot clustered points
plt.subplot(1, 2, 1)
plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], c=clustering_model.cluster_labels, cmap='viridis')
plt.title('Cluster Assignments')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Plot the dendrogram
plt.subplot(1, 2, 2)
clustering_model.plot_dendrogram()

# Adjust layout
plt.tight_layout()
plt.show()

import numpy as np

# Define the input feature matrix (each row is a sample, columns are features)
X = np.array([[2, 3], [3, 5], [5, 7], [7, 9]])  # 4 samples, 2 features
y = np.array([0, 0, 1, 1])  # Class labels (two classes: 0 and 1)

# Separate the data points based on class labels
class_0 = X[y == 0]  # Extract samples belonging to class 0
class_1 = X[y == 1]  # Extract samples belonging to class 1

# Compute the mean vectors for each class (column-wise mean)
mean_0 = np.mean(class_0, axis=0)  # Mean of class 0
mean_1 = np.mean(class_1, axis=0)  # Mean of class 1

# Compute the total mean (not used in this version but useful for reference)
mean_total = np.mean(X, axis=0)  # Overall mean of all samples

# Compute the within-class scatter matrix (S_W)
# S_W captures how much the data points of each class vary from their mean
S_W = np.dot((class_0 - mean_0).T, (class_0 - mean_0)) + np.dot((class_1 - mean_1).T, (class_1 - mean_1))

# Compute the between-class scatter matrix (S_B)
# S_B captures how much the class means differ from each other
mean_diff = (mean_0 - mean_1).reshape(-1, 1)  # Column vector of mean difference
S_B = np.dot(mean_diff, mean_diff.T)  # Outer product to form the matrix

# Solve the eigenvalue problem for inv(S_W) * S_B
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

# Choose the eigenvector corresponding to the largest eigenvalue
lda_direction = eig_vecs[:, np.argmax(eig_vals)]  # This is the optimal projection direction

# Project the original data points onto the LDA direction
X_lda_manual = np.dot(X, lda_direction)

# Print the projected data points
print("LDA result using manual matrix multiplication:\n", X_lda_manual)

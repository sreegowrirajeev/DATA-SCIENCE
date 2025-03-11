import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Define the input data (each row represents a sample, and columns are features)
X = np.array([[2, 3], [3, 5], [5, 7], [7, 9]])  # Feature matrix (2D points)
y = np.array([0, 0, 1, 1])  # Class labels (two classes: 0 and 1)

# Create an LDA model with 1 component
lda = LinearDiscriminantAnalysis(n_components=1)

# Fit the LDA model and transform the data onto the new axis
X_lda = lda.fit_transform(X, y)

# Print the transformed data points in the reduced dimension
print("LDA result using library method:\n", X_lda)

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initialize the Support Vector Machine (SVM) classifier.

        Parameters:
        - learning_rate: Learning rate for gradient descent
        - lambda_param: Regularization parameter (to avoid overfitting)
        - n_iters: Number of iterations for training loop
        """
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None    # Weight vector (to be learned)
        self.bias = None       # Bias term (to be learned)

    def fit(self, X_train, y_train):
        """
        Train the SVM model using the training data.

        Parameters:
        - X_train: Feature matrix of shape (n_samples, n_features)
        - y_train: Labels vector of shape (n_samples,)
        """
        num_samples, num_features = X_train.shape

        # Convert labels to {-1, 1} for hinge loss
        y_processed = np.where(y_train <= 0, -1, 1)

        # Initialize weights and bias to zero
        self.weights = np.zeros(num_features)
        self.bias = 0

        # Perform gradient descent
        for _ in range(self.n_iters):
            for idx, feature_vector in enumerate(X_train):
                condition = y_processed[idx] * (np.dot(feature_vector, self.weights) - self.bias) >= 1

                if condition:
                    # Correct classification: apply regularization update
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # Misclassified: update both weights and bias
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - y_processed[idx] * feature_vector)
                    self.bias -= self.learning_rate * y_processed[idx]

    def predict(self, X):
        """
        Predict class labels for given samples.

        Parameters:
        - X: Input feature matrix

        Returns:
        - Predicted class labels (-1 or 1)
        """
        linear_result = np.dot(X, self.weights) - self.bias
        return np.sign(linear_result)

    def decision_function(self, X):
        """
        Calculate distance of samples to the decision boundary.

        Parameters:
        - X: Input feature matrix

        Returns:
        - Distance values (real numbers)
        """
        return np.dot(X, self.weights) - self.bias


# Function to visualize decision boundary and margin
def plot_decision_boundary(model, X, y):
    # Plot data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)

    # Get axis limits
    axis = plt.gca()
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()

    # Create grid to evaluate decision function
    x_axis = np.linspace(x_limits[0], x_limits[1], 30)
    y_axis = np.linspace(y_limits[0], y_limits[1], 30)
    Y, X_grid = np.meshgrid(y_axis, x_axis)
    grid_points = np.vstack([X_grid.ravel(), Y.ravel()]).T
    Z = model.decision_function(grid_points).reshape(X_grid.shape)

    # Plot decision boundary and margins
    axis.contour(X_grid, Y, Z, colors='k', levels=[-1, 0, 1], alpha=0.6, linestyles=['--', '-', '--'])

    plt.title("SVM Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic binary classification data with 2 informative features
    features, labels = make_classification(n_samples=100, n_features=2, n_redundant=0,
                                           n_informative=2, random_state=1, n_clusters_per_class=1)

    # Convert label 0 to -1 (for SVM hinge loss)
    labels = np.where(labels == 0, -1, 1)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Create and train SVM model
    svm_model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm_model.fit(X_train, y_train)

    # Predict on test data
    test_predictions = svm_model.predict(X_test)

    # Evaluate model accuracy
    model_accuracy = np.mean(test_predictions == y_test)
    print(f"SVM Model Accuracy: {model_accuracy:.4f}")

    # Plot the decision boundary
    plot_decision_boundary(svm_model, X_train, y_train)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features='sqrt', random_state=None):
        """
        Initialize the Random Forest parameters.
        
        Parameters:
        - n_estimators: Number of decision trees
        - max_depth: Maximum depth of each decision tree
        - max_features: Number of features to consider at each split
        - random_state: Seed for random operations
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.decision_trees = []             # To store trained decision trees
        self.selected_feature_sets = []      # To store feature indices used for each tree

    def fit(self, features, labels):
        """
        Train the Random Forest model using the input data.
        
        Parameters:
        - features: Input feature matrix (samples × features)
        - labels: Target class labels
        """
        np.random.seed(self.random_state)             # Set random seed
        num_samples, num_total_features = features.shape  # Get data shape

        # Determine how many features to consider for each split
        if self.max_features == 'sqrt':
            num_features_to_select = int(np.sqrt(num_total_features))
        elif isinstance(self.max_features, int):
            num_features_to_select = self.max_features
        else:
            num_features_to_select = num_total_features

        self.decision_trees = []               # Reset list of trees
        self.selected_feature_sets = []        # Reset list of feature sets

        for _ in range(self.n_estimators):
            # Generate bootstrap sample
            sampled_features, sampled_labels = resample(features, labels, random_state=self.random_state)

            # Select random subset of features
            chosen_feature_indices = np.random.choice(num_total_features, num_features_to_select, replace=False)
            sampled_features_subset = sampled_features[:, chosen_feature_indices]

            # Train a decision tree classifier on the sampled data
            tree = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(sampled_features_subset, sampled_labels)

            # Store the trained tree and selected features
            self.decision_trees.append(tree)
            self.selected_feature_sets.append(chosen_feature_indices)

    def predict_proba(self, features):
        """
        Predict class probabilities by averaging the predictions from all trees.
        
        Parameters:
        - features: Input feature matrix
        
        Returns:
        - Average class probabilities
        """
        num_samples = features.shape[0]
        all_tree_predictions = []

        # Predict using each decision tree
        for tree, feature_indices in zip(self.decision_trees, self.selected_feature_sets):
            features_subset = features[:, feature_indices]          # Use only relevant features for this tree
            probabilities = tree.predict_proba(features_subset)     # Get class probabilities
            all_tree_predictions.append(probabilities)              # Store probabilities

        # Average probabilities across all trees
        average_probabilities = np.mean(all_tree_predictions, axis=0)
        return average_probabilities

    def predict(self, features):
        """
        Predict class labels based on the highest average probability.
        
        Parameters:
        - features: Input feature matrix
        
        Returns:
        - Predicted class labels
        """
        class_probabilities = self.predict_proba(features)        # Get class probabilities
        predictions = np.argmax(class_probabilities, axis=1)     # Choose the class with highest probability
        return predictions

# Example usage
if __name__ == "__main__":
    # Load the Iris dataset
    iris = load_iris()
    X_data, y_data = iris.data, iris.target

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

    # Create RandomForest instance with specified parameters
    custom_rf = RandomForest(n_estimators=100, max_depth=3, random_state=42)

    # Train the model on the training data
    custom_rf.fit(X_train, y_train)

    # Predict class labels on test data
    y_predictions = custom_rf.predict(X_test)

    # Calculate and display accuracy
    model_accuracy = accuracy_score(y_test, y_predictions)
    print(f"Custom Random Forest Accuracy: {model_accuracy:.4f}")

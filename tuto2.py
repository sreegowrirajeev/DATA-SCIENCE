import pandas as pd  # Import pandas for data handling
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for visualization
import seaborn as sns  # Import seaborn for enhanced visualizations
from sklearn.model_selection import train_test_split  # Import function for splitting data
from sklearn.linear_model import LogisticRegression  # Import logistic regression model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Import evaluation metrics

# Load dataset from the specified file path
dataset_path = "students_performance.csv"
data = pd.read_csv(dataset_path)

# Split dataset into predictor variables (features) and target variable (Pass/Fail outcome)
X = data[['Study Hours', 'Past Scores']]  # Features: Study hours and past scores
y = data['Pass']  # Target variable: 1 (Pass) or 0 (Fail)

# Split dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
model = LogisticRegression()

# Train the model using the training dataset
model.fit(X_train, y_train)

# Make predictions on the test dataset
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy
print("Accuracy:", accuracy)  # Print accuracy score
print("Classification Report:\n", classification_report(y_test, y_pred))  # Print classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))  # Print confusion matrix

# Visualize data distribution and classification
sns.scatterplot(x=data['Study Hours'], y=data['Past Scores'], hue=data['Pass'], palette='coolwarm')
plt.xlabel("Study Hours")  # Label x-axis
plt.ylabel("Past Scores")  # Label y-axis
plt.title("Logistic Regression Decision Boundary")  # Set plot title
plt.show()  # Display plot

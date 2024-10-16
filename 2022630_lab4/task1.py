from sklearn.datasets import load_wine
# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Load the sample dataset (Wine dataset)
wine = load_wine()
X = wine.data  # Features
y = wine.target  # Labels (target classes)

# Display dataset information
print("Dataset Information:")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {wine.feature_names}")
print(f"Target classes: {wine.target_names}")
print(f"Class distribution: {np.bincount(y)}\n")  # Shows how many samples per class


# Converting the Iris dataset to a DataFrame for better visualization
wine_df = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                       columns=wine['feature_names'] + ['target'])
# Mapping target values to class names (0 -> Setosa, 1 -> Versicolor, 2 -> Virginica)
wine_df['target'] = wine_df['target'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})
# Display the first few rows of the dataset
display(wine_df)

# Visualizing the data before classification (using first two features for 2D plot)
plt.figure(figsize=(8,6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.title('wine Dataset Visualization (Before Classification)')
plt.xlabel(wine.feature_names[0])  # Feature 1 (sepal length)
plt.ylabel(wine.feature_names[1])  # Feature 2 (sepal width)
colorbar=plt.colorbar(label='Classes')
colorbar.set_ticks([0, 1, 2])
colorbar.set_ticklabels(wine.target_names)  # Adding class labels to colorbar
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Implementing k-Nearest Neighbors with k=3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)  # Train the k-NN model on the training data

# Making predictions on the test set
y_pred = knn.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print accuracy and classification report
print(f"Accuracy of k-NN: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

# Visualizing the data after classification (using first two features for 2D plot)
plt.figure(figsize=(8,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title('wine Dataset Classification Visualization (After Classification)')
plt.xlabel(wine.feature_names[0])  # Feature 1 (sepal length)
plt.ylabel(wine.feature_names[1])  # Feature 2 (sepal width)
colorbar=plt.colorbar(label='Predicted Classes')
colorbar.set_ticks([0, 1, 2])
colorbar.set_ticklabels(wine.target_names)  # Adding class labels to colorbar
plt.show()

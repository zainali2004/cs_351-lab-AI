
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load and preprocess Wine dataset
wine = load_wine()
X_wine = wine.data
y_wine = to_categorical(wine.target)

scaler = StandardScaler()
X_wine = scaler.fit_transform(X_wine)

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Build the neural network for Wine dataset
model_wine = Sequential([
    Dense(8, input_shape=(X_wine.shape[1],), activation='relu'),
    Dense(16, activation='relu'),  # Second hidden layer
    Dense(3, activation='softmax')
])

model_wine.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history_wine = model_wine.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=8, verbose=0)

# Evaluate the model
test_loss_wine, test_accuracy_wine = model_wine.evaluate(X_test, y_test, verbose=0)

# Plot metrics
plot_metrics(history_wine, title="Wine Dataset")

# Confusion matrix for test predictions
y_pred_wine = model_wine.predict(X_test).argmax(axis=1)
y_true_wine = y_test.argmax(axis=1)
disp = ConfusionMatrixDisplay(confusion_matrix(y_true_wine, y_pred_wine), display_labels=wine.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Wine Dataset - Confusion Matrix")
plt.show()
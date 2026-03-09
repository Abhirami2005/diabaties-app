import pandas as pd
import pickle  # model save
from sklearn.linear_model import LogisticRegression  # ml model
from sklearn.model_selection import train_test_split  # split the data
from sklearn.metrics import accuracy_score  # accuracy finding

# Load dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data[['Glucose', 'BloodPressure', 'BMI', 'Age']]
y = data['Outcome']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save accuracy separately
with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy, 2)))

print("Model Saved Successfully!")
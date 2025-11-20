# train.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import skops.io as sio
from sklearn import set_config

# -----------------------------
# 1. Load and shuffle dataset
# -----------------------------
weather_df = pd.read_csv("Data/data.csv")
weather_df = weather_df.sample(frac=1)
print("Top 3 rows of the dataset:")
print(weather_df.head(3))

# -----------------------------
# 2. Train-test split
# -----------------------------
X = weather_df.drop("Rain", axis=1)
y = weather_df["Rain"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=125
)

# -----------------------------
# 3. Build pipeline
# -----------------------------
numerical_features = ['Temperature', 'Humidity', 'Wind_Speed', 'Cloud_Cover', 'Pressure']

preprocessor = ColumnTransformer(
    transformers=[
        ("num_pipeline",
         Pipeline([
             ('imputer', SimpleImputer(strategy="median")),
             ('scaler', StandardScaler())
         ]),
         numerical_features
        ),
    ]
)

full_pipeline = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125, n_jobs=-1)),
    ]
)

# -----------------------------
# 4. Train model
# -----------------------------
full_pipeline.fit(X_train, y_train)
print("Model training complete.")

# -----------------------------
# 5. Evaluate model
# -----------------------------
predictions = full_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="macro")
print("Accuracy:", str(round(accuracy, 2) * 100) + "%", "F1:", round(f1, 2))

# Save metrics
os.makedirs("Results", exist_ok=True)
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f"\nAccuracy = {round(accuracy, 2)}, F1 Score = {round(f1, 2)}.")

# Confusion matrix
cm = confusion_matrix(y_test, predictions, labels=full_pipeline.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=full_pipeline.classes_)
disp.plot()
plt.savefig("Results/model_results.png", dpi=120)
plt.close()

# -----------------------------
# 6. Save model
# -----------------------------
os.makedirs("Model", exist_ok=True)
sio.dump(full_pipeline, "Model/weather_pipeline.skops")
print("Pipeline saved to Model/weather_pipeline.skops")

# -----------------------------
# 7. Load model
# -----------------------------
loaded_model = sio.load(
    "Model/weather_pipeline.skops",
    trusted=['numpy.dtype']  # trusted types list
)
print("Loaded model type:", type(loaded_model))

# -----------------------------
# 8. Sample predictions
# -----------------------------
sample_prediction = loaded_model.predict(X_test[:5])
print("Sample predictions:", sample_prediction)

# -----------------------------
# 9. Display pipeline diagram
# -----------------------------
set_config(display='diagram')
loaded_model

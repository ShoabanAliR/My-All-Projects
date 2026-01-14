import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/student_data.csv")
df = df.drop("study_hours", axis= 1)


# Features & target
X = df.drop("final_grade", axis=1)
y = df["final_grade"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("student_model.pkl", "wb"))

print("Model Trained & Saved Successfully")

import pandas as pd
from sklearn.linear_model import LinearRegression
import json

# Load dataset
data = pd.read_csv("student_grade_prediction_dataset.csv")  # make path simple for deployment

# Features and target
X = data[['StudyHours', 'PreviousGrade']]
y = data['FinalGrade']

# Train model
model = LinearRegression()
model.fit(X, y)

# Extract coefficients
intercept = model.intercept_
coef = model.coef_

# Save intercept and coefficients to config.json
model_data = {
    'intercept': intercept,
    'coef_hours': coef[0],
    'coef_prev': coef[1]
}

with open('config.json', 'w') as f:
    json.dump(model_data, f)

print("Model parameters saved to config.json")


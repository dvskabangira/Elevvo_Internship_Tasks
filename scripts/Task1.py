
# Dataset (Recommended): Student Performance Factors (Kaggle)
# Build a model to predict students' exam scores based on their study hour
# Perform data cleaning and basic visualization to understand the datase
# Split the dataset into training and testing set
# Train a linear regression model to estimate final score
# Visualize predictions and evaluate model performance

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#load data
data = pd.read_csv("/workspaces/Elevvo_Task1-Student_score_predicition/Data/StudentPerformanceFactors.csv")
#print(data.head(5))
#print(data.describe())

#Data cleaning

#print(data.isnull().sum())
df = data.dropna(subset=['Teacher_Quality','Parental_Education_Level','Distance_from_Home'])

#Data preprocessing
X = df['Hours_Studied']
X = df[['Hours_Studied']]
#print(X.shape)

y = df['Exam_Score']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=45)

#Building & Training the linear model
model = LinearRegression()
model.fit(X_train,y_train)

# Predict on test set
y_pred = model.predict(X_test)
print("Predicted scores:", y_pred)


# Model performance
print("Slope (m):", model.coef_[0])
print("Intercept (c):", model.intercept_)
print("R² Score:", r2_score(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# Plot results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Study Hours")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Score Prediction")
plt.legend()
plt.show()

# Predict for a new student
new_hours = np.array([[7.5]])
predicted_score = model.predict(new_hours)
print(f"Predicted score for 7.5 hours of study: {predicted_score[0]:.2f}")

# Student Performance Factors (Kaggle Dataset)
# Build a model to predict students' exam scores based on their study hour
# Perform data cleaning and basic visualization to understand the datase
# Split the dataset into training and testing set
# Train a linear regression model to estimate final score
# Visualize predictions and evaluate model performance

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#load data & simple visualization
data = pd.read_csv("/workspaces/Elvvo_Task1-Student_score_predicition/Data/StudentPerformanceFactors.csv")
print(data.head(5))
print(data.describe())

#Data cleaning

print(data.isnull().sum())
df = data.dropna(subset=['Teacher_Quality','Parental_Education_Level','Distance_from_Home'])


#Data preprocessing

cdf = df[['Hours_Studied','Sleep_Hours','Access_to_Resources','Exam_Score']]
print(cdf.sample(2))

X = cdf.Hours_Studied.to_numpy()
print(X.shape)
y = cdf.Exam_Score.to_numpy()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=45)

print(np.shape(X_train)), np.shape(y_train), print(type(X_train))

#Building & Training the linear model
regressor = LinearRegression()
regressor.fit(X_train.reshape(-1,1), y_train)

#print coeffients
print("Slope (m):", regressor.coef_[0])
print("Intercept (c):", regressor.intercept_)


# visualize model outputs
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, 'r')
plt.xlabel('Hours_Studied')
plt.ylabel('Exam_Score')
plt.show()


# Predict on test set
y_pred = regressor.predict(X_test.reshape(-1,1))
print("Predicted scores:", y_pred)


# Model Evaluation

print("R² Score:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))


# Predict for a new student
new_hours = np.array([[7.5]])
predicted_score = regressor.predict(new_hours)
print(f"Predicted score for 7.5 hours of study: {predicted_score[0]:.2f}")




##Polynomial Regression


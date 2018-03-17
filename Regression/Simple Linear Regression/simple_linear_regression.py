# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Fitting the training set to the Simple Linear Regression
X_train = np.reshape(X_train,(-1,2))
y_train = np.reshape(y_train,(-1,2))
X_test = np.reshape(X_test,(-1,2))
y_test = np.reshape(y_test,(-1,2))
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Predicting the Test Set results
y_pred = regressor.predict(X_test)

#Visualizing the Training set results
plt.scatter(X_train,y_train,color = "red")
plt.plot(X_train, regressor.predict(X_train),color="blue")
plt.title("Salary VS Experience(Training Set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()

#Visualizing the Test set results
plt.scatter(X_test,y_test,color = "red")
plt.plot(X_train, regressor.predict(X_train),color="blue")
plt.title("Salary VS Experience(Test set)")
plt.xlabel("Years Of Experience")
plt.ylabel("Salary")
plt.show()
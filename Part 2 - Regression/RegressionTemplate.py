# Regression Template


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

file_name = 'Position_Salaries.csv'
test_set_size = 0.2

# Importing the dataset
dataset = pd.read_csv(file_name)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_set_size, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting the Regression Model

# Predicting a new result with the Regression
y_pred = regressor.predict(6.5)

# Visualising the Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth of Bluff (Regression Model)')
plt.xlabel('1')
plt.ylabel('2')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_grid), color = 'blue')
plt.title('Truth of Bluff (Regression Model)')
plt.xlabel('1')
plt.ylabel('2')
plt.show()